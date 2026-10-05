// Allocation of KV-cache blocks, with automatic prefix caching.
//
// Every block holds kBlockSize tokens. A block that is completely filled by a
// prompt (or by generated text) is "sealed": its contents can never change, so
// it is registered under a hash of its tokens chained with the hash of the
// block before it - a key for the whole prefix up to and including it.
//
// A later sequence whose prompt starts with the same tokens finds those
// blocks and shares them (reference counts), skipping their computation.
// When the last user of a sealed block lets go, the block is not wiped: it
// waits in an LRU list, still findable, until memory is needed - so a system
// prompt reused across requests stays cached as long as space allows.
#pragma once

#include <cstdint>
#include <list>
#include <span>
#include <unordered_map>
#include <vector>

#include "backend/backend.hpp"

namespace ember {

class BlockManager {
public:
    BlockManager(int num_blocks, bool prefix_caching);

    int num_blocks() const { return static_cast<int>(blocks_.size()); }
    // Blocks that can be handed out now: free, or cached but unused.
    int available() const { return static_cast<int>(free_.size() + evictable_.size()); }
    int used() const { return num_blocks() - available(); }
    int cached() const { return static_cast<int>(by_hash_.size()); }

    // Takes a fresh block; -1 if none is available. Evicts the least recently
    // used cached block when there is no free one.
    int allocate();
    // Drops one reference; at zero the block becomes free (or evictable if sealed).
    void release(int block);
    void release_all(std::span<const int> blocks);

    // Hash of a full block of tokens following a prefix with hash `parent` (0 for the first block).
    static uint64_t block_hash(uint64_t parent, std::span<const int32_t> tokens);

    // Finds the longest run of cached blocks matching the start of `tokens`
    // (only whole blocks), takes a reference to each, and returns them.
    std::vector<int> match_prefix(std::span<const int32_t> tokens);

    // Declares `block` full with `tokens` after a prefix with hash `parent`.
    // Returns the block's hash (to chain the next one). If an identical block is
    // already cached, nothing changes (the duplicate stays private).
    uint64_t seal(int block, uint64_t parent, std::span<const int32_t> tokens);

    bool prefix_caching() const { return prefix_caching_; }
    int refcount(int block) const { return blocks_[static_cast<size_t>(block)].refs; }

    // Statistics.
    uint64_t hit_tokens() const { return hit_tokens_; }
    uint64_t lookup_tokens() const { return lookup_tokens_; }

    // Debug check: every block is exactly one of free / evictable / referenced,
    // and the hash index agrees with the blocks. Returns a description of the
    // first problem, or an empty string.
    std::string check() const;

    // Per-block state for the dashboard: 0 free, 1 cached (unused), 2 in use.
    std::vector<uint8_t> states() const;

private:
    struct Block {
        int refs = 0;
        bool sealed = false;
        uint64_t hash = 0;
        std::vector<int32_t> tokens;           // copy of a sealed block's tokens, to rule out hash collisions
        uint64_t parent = 0;
        std::list<int>::iterator lru;          // position in evictable_ when refs == 0 && sealed
    };

    void unseal(int block);

    std::vector<Block> blocks_;
    std::vector<int> free_;                          // never-sealed free blocks (stack)
    std::list<int> evictable_;                       // sealed, unreferenced, oldest first
    std::unordered_map<uint64_t, int> by_hash_;
    bool prefix_caching_;
    uint64_t hit_tokens_ = 0, lookup_tokens_ = 0;
};

}  // namespace ember
