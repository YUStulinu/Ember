#include "server/http.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <format>

#include "common/error.hpp"
#include "common/log.hpp"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
using sock_t = SOCKET;
static const sock_t kBadSocket = INVALID_SOCKET;
static void close_socket(sock_t s) { closesocket(s); }
static constexpr int kSendFlags = 0;
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
using sock_t = int;
static const sock_t kBadSocket = -1;
static void close_socket(sock_t s) { ::close(s); }
#ifdef MSG_NOSIGNAL
static constexpr int kSendFlags = MSG_NOSIGNAL;  // a write to a closed socket must not raise SIGPIPE
#else
static constexpr int kSendFlags = 0;             // macOS: SO_NOSIGPIPE is set on each socket instead
#endif
#endif

namespace ember::http {

namespace {

void net_init() {
#ifdef _WIN32
    static bool done = [] {
        WSADATA w;
        if (WSAStartup(MAKEWORD(2, 2), &w) != 0) fail("WSAStartup failed");
        return true;
    }();
    (void)done;
#endif
}

std::string lower(std::string s) {
    for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string trim(const std::string &s) {
    size_t b = 0, e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t')) b++;
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r')) e--;
    return s.substr(b, e - b);
}

}  // namespace

// One client connection: a socket and its unread input.
class Connection {
public:
    Connection(sock_t s, std::string remote) : sock_(s), remote_(std::move(remote)) {}
    ~Connection() { close(); }

    void close() {
        sock_t s = sock_.exchange(kBadSocket);
        if (s != kBadSocket) close_socket(s);
    }
    void shutdown_now() {
        sock_t s = sock_.load();
        if (s != kBadSocket) ::shutdown(s, 2);
    }

    bool send_all(const char *data, size_t n) {
        sock_t s = sock_.load();
        while (n > 0 && s != kBadSocket) {
            int sent = ::send(s, data, static_cast<int>(std::min<size_t>(n, 1 << 20)), kSendFlags);
            if (sent <= 0) {
                broken_ = true;
                return false;
            }
            data += sent;
            n -= static_cast<size_t>(sent);
        }
        return !broken_ && s != kBadSocket;
    }
    bool send_all(const std::string &s) { return send_all(s.data(), s.size()); }

    // Reads more input; false on EOF or error.
    bool fill() {
        char tmp[16384];
        int got = ::recv(sock_.load(), tmp, sizeof tmp, 0);
        if (got <= 0) return false;
        buf_.append(tmp, static_cast<size_t>(got));
        return true;
    }

    // Parses one request. False when the connection closed (or sent garbage).
    bool read_request(Request &req, size_t max_body, int &error_status) {
        error_status = 0;
        size_t end;
        while ((end = buf_.find("\r\n\r\n")) == std::string::npos) {
            if (buf_.size() > 64 * 1024) {
                error_status = 431;
                return false;
            }
            if (!fill()) return false;
        }
        std::string head = buf_.substr(0, end);
        buf_.erase(0, end + 4);
        size_t line_end = head.find("\r\n");
        std::string first = head.substr(0, line_end);
        size_t a = first.find(' '), b = first.rfind(' ');
        if (a == std::string::npos || b == a) {
            error_status = 400;
            return false;
        }
        req = Request{};
        req.method = first.substr(0, a);
        std::string target = first.substr(a + 1, b - a - 1);
        req.version = first.substr(b + 1);
        size_t q = target.find('?');
        req.path = url_decode(target.substr(0, q));
        if (q != std::string::npos) req.query = target.substr(q + 1);
        req.remote = remote_;
        size_t pos = line_end == std::string::npos ? head.size() : line_end + 2;
        while (pos < head.size()) {
            size_t e = head.find("\r\n", pos);
            if (e == std::string::npos) e = head.size();
            std::string line = head.substr(pos, e - pos);
            size_t colon = line.find(':');
            if (colon != std::string::npos) req.headers[lower(trim(line.substr(0, colon)))] = trim(line.substr(colon + 1));
            pos = e + 2;
        }
        if (req.header("transfer-encoding").size()) {
            error_status = 411;  // chunked request bodies: not supported
            return false;
        }
        size_t length = 0;
        if (auto cl = req.header("content-length"); !cl.empty()) {
            try {
                length = static_cast<size_t>(std::stoull(cl));
            } catch (...) {
                error_status = 400;
                return false;
            }
        }
        if (length > max_body) {
            error_status = 413;
            return false;
        }
        while (buf_.size() < length)
            if (!fill()) return false;
        req.body = buf_.substr(0, length);
        buf_.erase(0, length);
        return true;
    }

    // True if the peer closed its end (checked without blocking).
    bool peer_closed() {
        if (broken_) return true;
        sock_t s = sock_.load();
        if (s == kBadSocket) return true;
#ifdef _WIN32
        fd_set set;
        FD_ZERO(&set);
        FD_SET(s, &set);
        timeval tv{0, 0};
        if (select(0, &set, nullptr, nullptr, &tv) <= 0) return false;
#else
        pollfd p{s, POLLIN, 0};
        if (poll(&p, 1, 0) <= 0) return false;
        if (p.revents & (POLLHUP | POLLERR)) return true;
#endif
        char c;
        int got = ::recv(s, &c, 1, MSG_PEEK);
        return got == 0;  // readable with no data: EOF (pipelined data would be > 0)
    }

    bool keep_alive = true;

private:
    std::atomic<sock_t> sock_;
    std::string remote_;
    std::string buf_;
    bool broken_ = false;
};

std::string Request::header(const std::string &name, const std::string &def) const {
    auto it = headers.find(name);
    return it == headers.end() ? def : it->second;
}

const char *status_text(int s) {
    switch (s) {
        case 200: return "OK";
        case 204: return "No Content";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 411: return "Length Required";
        case 413: return "Payload Too Large";
        case 429: return "Too Many Requests";
        case 431: return "Request Header Fields Too Large";
        case 500: return "Internal Server Error";
        case 503: return "Service Unavailable";
        default: return "Unknown";
    }
}

std::string url_decode(const std::string &s) {
    std::string out;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '%' && i + 2 < s.size() && std::isxdigit(static_cast<unsigned char>(s[i + 1])) &&
            std::isxdigit(static_cast<unsigned char>(s[i + 2]))) {
            out += static_cast<char>(std::stoi(s.substr(i + 1, 2), nullptr, 16));
            i += 2;
        } else {
            out += s[i] == '+' ? ' ' : s[i];
        }
    }
    return out;
}

static std::string head_of(int status, const std::string &type, const std::vector<std::pair<std::string, std::string>> &extra,
                           bool keep_alive) {
    std::string h = std::format("HTTP/1.1 {} {}\r\nContent-Type: {}\r\nAccess-Control-Allow-Origin: *\r\n", status,
                                status_text(status), type);
    for (const auto &[k, v] : extra) h += k + ": " + v + "\r\n";
    h += keep_alive ? "Connection: keep-alive\r\n" : "Connection: close\r\n";
    return h;
}

bool Response::respond(int status, const std::string &type, const std::string &body,
                       const std::vector<std::pair<std::string, std::string>> &extra) {
    if (state_ != State::idle) return false;
    state_ = State::done;
    std::string out = head_of(status, type, extra, conn_.keep_alive);
    out += std::format("Content-Length: {}\r\n\r\n", body.size());
    out += body;
    return conn_.send_all(out);
}

bool Response::start_stream(int status, const std::string &type, const std::vector<std::pair<std::string, std::string>> &extra) {
    if (state_ != State::idle) return false;
    state_ = State::streaming;
    std::string out = head_of(status, type, extra, conn_.keep_alive);
    out += "Cache-Control: no-cache\r\nX-Accel-Buffering: no\r\nTransfer-Encoding: chunked\r\n\r\n";
    return conn_.send_all(out);
}

bool Response::write(const std::string &data) {
    if (state_ != State::streaming) return false;
    if (data.empty()) return true;
    return conn_.send_all(std::format("{:x}\r\n", data.size()) + data + "\r\n");
}

bool Response::finish() {
    if (state_ != State::streaming) return state_ == State::done;
    state_ = State::done;
    return conn_.send_all("0\r\n\r\n");
}

bool Response::client_gone() const { return conn_.peer_closed(); }

Server::Server() { net_init(); }

Server::~Server() { stop(); }

void Server::route(const std::string &method, const std::string &path, Handler h) {
    routes_.push_back({method, path, std::move(h)});
}

void Server::start(const std::string &host, int port) {
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    std::string port_s = std::to_string(port);
    if (getaddrinfo(host.empty() ? nullptr : host.c_str(), port_s.c_str(), &hints, &res) != 0 || !res)
        fail("cannot resolve {}:{}", host, port);
    sock_t s = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (s == kBadSocket) {
        freeaddrinfo(res);
        fail("cannot create a socket");
    }
#ifndef _WIN32
    int yes = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof yes);
#endif
    if (::bind(s, res->ai_addr, static_cast<int>(res->ai_addrlen)) != 0 || ::listen(s, 128) != 0) {
        freeaddrinfo(res);
        close_socket(s);
        fail("cannot listen on {}:{} (is the port in use?)", host, port);
    }
    freeaddrinfo(res);
    sockaddr_storage addr{};
    socklen_t len = sizeof addr;
    getsockname(s, reinterpret_cast<sockaddr *>(&addr), &len);
    port_ = ntohs(addr.ss_family == AF_INET6 ? reinterpret_cast<sockaddr_in6 *>(&addr)->sin6_port
                                             : reinterpret_cast<sockaddr_in *>(&addr)->sin_port);
    listener_ = static_cast<intptr_t>(s);
    stopping_ = false;
    acceptor_ = std::thread([this] { accept_loop(); });
}

void Server::stop() {
    if (listener_ == -1) return;
    stopping_ = true;
    close_socket(static_cast<sock_t>(listener_));
    listener_ = -1;
    if (acceptor_.joinable()) acceptor_.join();
    {
        std::lock_guard<std::mutex> lock(conns_mutex_);
        for (auto &w : conns_)
            if (auto c = w.lock()) c->shutdown_now();
    }
    for (int i = 0; i < 300 && active_.load() > 0; i++) std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

void Server::accept_loop() {
    while (!stopping_) {
        sockaddr_storage addr{};
        socklen_t len = sizeof addr;
        sock_t c = ::accept(static_cast<sock_t>(listener_), reinterpret_cast<sockaddr *>(&addr), &len);
        if (c == kBadSocket) {
            if (stopping_) return;
            continue;
        }
        int one = 1;
        setsockopt(c, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char *>(&one), sizeof one);  // stream tokens promptly
#ifdef SO_NOSIGPIPE
        setsockopt(c, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
        char ip[INET6_ADDRSTRLEN] = "?";
        if (addr.ss_family == AF_INET) inet_ntop(AF_INET, &reinterpret_cast<sockaddr_in *>(&addr)->sin_addr, ip, sizeof ip);
        else inet_ntop(AF_INET6, &reinterpret_cast<sockaddr_in6 *>(&addr)->sin6_addr, ip, sizeof ip);
        auto conn = std::make_shared<Connection>(c, ip);
        {
            std::lock_guard<std::mutex> lock(conns_mutex_);
            conns_.erase(std::remove_if(conns_.begin(), conns_.end(), [](auto &w) { return w.expired(); }), conns_.end());
            conns_.push_back(conn);
        }
        active_++;
        std::thread([this, conn] {
            serve(conn);
            active_--;
        }).detach();
    }
}

void Server::serve(std::shared_ptr<Connection> c) {
    while (!stopping_) {
        Request req;
        int error = 0;
        if (!c->read_request(req, max_body, error)) {
            if (error) {
                c->keep_alive = false;
                Response r(*c);
                r.respond(error, "application/json", std::format(R"({{"error":"{}"}})", status_text(error)));
            }
            break;
        }
        const std::string conn_hdr = lower(req.header("connection"));
        c->keep_alive = req.version == "HTTP/1.1" ? conn_hdr != "close" : conn_hdr == "keep-alive";
        Response res(*c);
        try {
            dispatch(req, res);
        } catch (const std::exception &e) {
            log::warn("{} {}: {}", req.method, req.path, e.what());
            if (!res.sent()) {
                std::string msg = e.what();
                for (char &ch : msg)
                    if (ch == '"' || ch == '\\') ch = '\'';
                res.respond(500, "application/json", std::format(R"({{"error":{{"message":"{}"}}}})", msg));
            } else {
                res.finish();
            }
        }
        if (!res.sent()) res.respond(500, "application/json", R"({"error":"no response"})");
        if (!c->keep_alive) break;
    }
    c->close();
}

void Server::dispatch(const Request &req, Response &res) {
    if (req.method == "OPTIONS") {  // CORS preflight from browsers
        res.respond(204, "text/plain", "",
                    {{"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
                     {"Access-Control-Allow-Headers", "Content-Type, Authorization, x-api-key, anthropic-version"}});
        return;
    }
    bool path_known = false;
    for (const auto &r : routes_) {
        if (r.path != req.path) continue;
        path_known = true;
        if (r.method == "*" || r.method == req.method) {
            r.handler(req, res);
            return;
        }
    }
    if (path_known) {
        res.respond(405, "application/json", R"({"error":"method not allowed"})");
        return;
    }
    if (fallback_) {
        fallback_(req, res);
        return;
    }
    res.respond(404, "application/json", R"({"error":"not found"})");
}

}  // namespace ember::http
