// A small HTTP/1.1 server: enough for a JSON API with streaming responses.
//
// One thread per connection (connections are few and long-lived: a client
// waiting for tokens), keep-alive, Content-Length request bodies, and chunked
// responses for Server-Sent Events. The GPU never waits for the network: the
// engine runs on its own thread and hands tokens to these threads.
#pragma once

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ember::http {

struct Request {
    std::string method, path, query, version;
    std::map<std::string, std::string> headers;  // names lower-cased
    std::string body;
    std::string remote;

    std::string header(const std::string &name, const std::string &def = "") const;
};

class Connection;

// Writes one response. Either respond() once, or start_stream() and then
// write() pieces (sent as HTTP chunks) until finish(). A write returns false
// once the client has gone away.
class Response {
public:
    explicit Response(Connection &c) : conn_(c) {}
    bool respond(int status, const std::string &content_type, const std::string &body,
                 const std::vector<std::pair<std::string, std::string>> &extra_headers = {});
    bool start_stream(int status, const std::string &content_type,
                      const std::vector<std::pair<std::string, std::string>> &extra_headers = {});
    bool write(const std::string &data);
    bool finish();
    bool sent() const { return state_ != State::idle; }
    bool client_gone() const;

private:
    enum class State { idle, streaming, done };
    Connection &conn_;
    State state_ = State::idle;
};

using Handler = std::function<void(const Request &, Response &)>;

class Server {
public:
    Server();
    ~Server();

    // Exact-path routes; "*" as method matches any.
    void route(const std::string &method, const std::string &path, Handler h);
    void set_fallback(Handler h) { fallback_ = std::move(h); }

    // Binds and starts accepting on a background thread; throws on failure.
    void start(const std::string &host, int port);
    void stop();
    int port() const { return port_; }

    size_t max_body = 16u << 20;

private:
    void accept_loop();
    void serve(std::shared_ptr<Connection> c);
    void dispatch(const Request &req, Response &res);

    struct Route {
        std::string method, path;
        Handler handler;
    };
    std::vector<Route> routes_;
    Handler fallback_;
    intptr_t listener_ = -1;
    int port_ = 0;
    std::atomic<bool> stopping_{false};
    std::thread acceptor_;
    std::mutex conns_mutex_;
    std::vector<std::weak_ptr<Connection>> conns_;
    std::atomic<int> active_{0};
};

const char *status_text(int status);
std::string url_decode(const std::string &s);

}  // namespace ember::http
