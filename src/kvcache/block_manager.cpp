#include "kvcache/block_manager.hpp"

#include <format>

#include "common/error.hpp"

namespace ember {

BlockManager::BlockManager(int num_blocks, bool prefix_caching) : blocks_(static_cast<size_t>(num_blocks)), prefix_caching_(prefix_caching) {
    EMBER_CHECK(num_blocks > 0, "the KV cache has no blocks");
    free_.reserve(static_cast<size_t>(num_blocks));
    for (int b = num_blocks - 1; b >= 0; b--) free_.push_back(b);  // hand out low numbers first
}

uint64_t BlockManager::block_hash(uint64_t parent, std::span<const int32_t> tokens) {
    // FNV-1a over the parent hash and the tokens, then a strong final mix.
    uint64_t h = 0xcbf29ce484222325ull ^ parent;
    for (int32_t t : tokens) {
        h ^= static_cast<uint32_t>(t);
        h *= 0x100000001b3ull;
    }
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ull;
    h ^= h >> 33;
    return h ? h : 1;  // 0 means "no parent"
}

int BlockManager::allocate() {
    int b;
    if (!free_.empty()) {
        b = free_.back();
        free_.pop_back();
    } else if (!evictable_.empty()) {
        b = evictable_.front();
        evictable_.pop_front();
        unseal(b);
    } else {
        return -1;
    }
    blocks_[static_cast<size_t>(b)].refs = 1;
    return b;
}

void BlockManager::unseal(int b) {
    Block &blk = blocks_[static_cast<size_t>(b)];
    if (!blk.sealed) return;
    auto it = by_hash_.find(blk.hash);
    if (it != by_hash_.end() && it->second == b) by_hash_.erase(it);
    blk.sealed = false;
    blk.hash = 0;
    blk.tokens.clear();
}

void BlockManager::release(int b) {
    Block &blk = blocks_.at(static_cast<size_t>(b));
    EMBER_CHECK(blk.refs > 0, "internal: releasing free block {}", b);
    if (--blk.refs > 0) return;
    if (blk.sealed && prefix_caching_ && by_hash_.count(blk.hash) && by_hash_[blk.hash] == b) {
        evictable_.push_back(b);
        blk.lru = std::prev(evictable_.end());
    } else {
        unseal(b);
        free_.push_back(b);
    }
}

void BlockManager::release_all(std::span<const int> blocks) {
    // Release in reverse, so a sequence's last blocks become the most likely to be evicted.
    for (size_t i = blocks.size(); i-- > 0;) release(blocks[i]);
}

std::vector<int> BlockManager::match_prefix(std::span<const int32_t> tokens) {
    std::vector<int> out;
    if (!prefix_caching_) return out;
    uint64_t parent = 0;
    const size_t full = tokens.size() / kBlockSize;
    lookup_tokens_ += tokens.size();
    for (size_t i = 0; i < full; i++) {
        auto chunk = tokens.subspan(i * kBlockSize, kBlockSize);
        uint64_t h = block_hash(parent, chunk);
        auto it = by_hash_.find(h);
        if (it == by_hash_.end()) break;
        Block &blk = blocks_[static_cast<size_t>(it->second)];
        if (blk.parent != parent || !std::equal(chunk.begin(), chunk.end(), blk.tokens.begin())) break;  // collision
        if (blk.refs == 0) evictable_.erase(blk.lru);
        blk.refs++;
        out.push_back(it->second);
        parent = h;
    }
    hit_tokens_ += out.size() * kBlockSize;
    return out;
}

uint64_t BlockManager::seal(int b, uint64_t parent, std::span<const int32_t> tokens) {
    EMBER_CHECK(tokens.size() == kBlockSize, "internal: sealing a block with {} tokens", tokens.size());
    uint64_t h = block_hash(parent, tokens);
    if (!prefix_caching_) return h;
    Block &blk = blocks_[static_cast<size_t>(b)];
    if (blk.sealed) return blk.hash;
    if (by_hash_.count(h)) return h;  // an identical block is already cached; this one stays private
    blk.sealed = true;
    blk.hash = h;
    blk.parent = parent;
    blk.tokens.assign(tokens.begin(), tokens.end());
    by_hash_[h] = b;
    return h;
}

std::string BlockManager::check() const {
    std::vector<int> seen(blocks_.size(), 0);
    for (int b : free_) {
        if (blocks_[static_cast<size_t>(b)].refs != 0) return std::format("free block {} has references", b);
        if (blocks_[static_cast<size_t>(b)].sealed) return std::format("free block {} is sealed", b);
        seen[static_cast<size_t>(b)]++;
    }
    for (int b : evictable_) {
        const Block &blk = blocks_[static_cast<size_t>(b)];
        if (blk.refs != 0 || !blk.sealed) return std::format("evictable block {} is in use or unsealed", b);
        seen[static_cast<size_t>(b)]++;
    }
    for (size_t b = 0; b < blocks_.size(); b++) {
        if (blocks_[b].refs > 0) seen[b]++;
        if (seen[b] != 1) return std::format("block {} is accounted for {} times", b, seen[b]);
    }
    for (const auto &[h, b] : by_hash_) {
        const Block &blk = blocks_[static_cast<size_t>(b)];
        if (!blk.sealed || blk.hash != h) return std::format("hash index points to block {} with a different hash", b);
    }
    return {};
}

std::vector<uint8_t> BlockManager::states() const {
    std::vector<uint8_t> s(blocks_.size(), 0);
    for (size_t b = 0; b < blocks_.size(); b++) s[b] = blocks_[b].refs > 0 ? 2 : 0;
    for (int b : evictable_) s[static_cast<size_t>(b)] = 1;
    return s;
}

}  // namespace ember
