/**
 * Byte pair encoding utilities.
 *
 * Ported from: https://github.com/openai/gpt-2/blob/master/src/encoder.py
 *
 * Usage (Node.js ESM):
 *
 *   import { getEncoderFromFiles, getEncoder } from "./encoder.mjs";
 *
 *   // Option A – load from the model directory (mirrors the Python API)
 *   const enc = await getEncoderFromFiles("./models/gpt2");
 *
 *   // Option B – pass file contents directly
 *   const enc = getEncoder(encoderJsonString, vocabBpeString);
 *
 *   const tokens = enc.encode("Hello world!");
 *   console.log(tokens);               // [15496, 995, 0]
 *   console.log(enc.decode(tokens));   // "Hello world!"
 */

import { readFile } from "node:fs/promises";
import { join }      from "node:path";

// ─── bytes_to_unicode ────────────────────────────────────────────────────────

let _bytesToUnicodeCache = null;

/**
 * Returns a mapping from UTF-8 byte values → single unicode characters.
 * Mirrors the Python bytes_to_unicode() with @lru_cache.
 */
function bytesToUnicode() {
  if (_bytesToUnicodeCache) return _bytesToUnicodeCache;

  const bs = [
    ...range(/* '!' */ 33,  /* '~' */ 126 + 1),
    ...range(/* '¡' */ 161, /* '¬' */ 172 + 1),
    ...range(/* '®' */ 174, /* 'ÿ' */ 255 + 1),
  ];
  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }

  const map = {};
  for (let i = 0; i < bs.length; i++) {
    map[bs[i]] = String.fromCodePoint(cs[i]);
  }
  _bytesToUnicodeCache = map;
  return map;
}

// ─── get_pairs ───────────────────────────────────────────────────────────────

/**
 * Return the set of adjacent-symbol pairs in a word (array of strings).
 * Pairs are stored as two-element arrays; a Map keyed by JSON is used so
 * Set-like deduplication works on value equality.
 *
 * @param   {string[]} word
 * @returns {Map<string, [string, string]>}  key → pair
 */
function getPairs(word) {
  const pairs = new Map();
  let prev = word[0];
  for (let i = 1; i < word.length; i++) {
    const cur = word[i];
    const key = JSON.stringify([prev, cur]);
    pairs.set(key, [prev, cur]);
    prev = cur;
  }
  return pairs;
}

// ─── Encoder ─────────────────────────────────────────────────────────────────

class Encoder {
  /**
   * @param {Record<string, number>} encoder   - token string → token id
   * @param {[string, string][]}     bpeMerges - ordered list of merge pairs
   * @param {string}                 errors    - TextDecoder error mode
   */
  constructor(encoder, bpeMerges, errors = "replace") {
    this.encoder = encoder;
    this.decoder = Object.fromEntries(
      Object.entries(encoder).map(([k, v]) => [v, k])
    );
    this.errors = errors;

    this.byteEncoder = bytesToUnicode();
    this.byteDecoder = Object.fromEntries(
      Object.entries(this.byteEncoder).map(([byteVal, char]) => [char, Number(byteVal)])
    );

    // Map from JSON-serialised pair → merge rank (lower = higher priority)
    this.bpeRanks = new Map(
      bpeMerges.map((pair, i) => [JSON.stringify(pair), i])
    );

    this.cache = new Map();

    // Same pattern as the Python original (minus the IGNORECASE note).
    // Uses the Unicode property escapes supported natively in Node ≥ 10.
    this.pat = /('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)/gu;
  }

  // ── bpe ──────────────────────────────────────────────────────────────────

  /**
   * Apply BPE to a single (byte-mapped) token string.
   * @param   {string} token
   * @returns {string}  space-separated BPE pieces
   */
  bpe(token) {
    if (this.cache.has(token)) return this.cache.get(token);

    let word = [...token];   // split into individual Unicode code points

    if (word.length === 1) {
      this.cache.set(token, token);
      return token;
    }

    let pairs = getPairs(word);

    while (true) {
      // Find the bigram with the lowest merge rank
      let bestKey   = null;
      let bestPair  = null;
      let bestRank  = Infinity;

      for (const [key, pair] of pairs) {
        const rank = this.bpeRanks.has(key) ? this.bpeRanks.get(key) : Infinity;
        if (rank < bestRank) {
          bestRank = rank;
          bestKey  = key;
          bestPair = pair;
        }
      }

      if (bestKey === null || !this.bpeRanks.has(bestKey)) break;

      const [first, second] = bestPair;
      const newWord = [];
      let i = 0;

      while (i < word.length) {
        const j = word.indexOf(first, i);
        if (j === -1) {
          newWord.push(...word.slice(i));
          break;
        }
        newWord.push(...word.slice(i, j));
        i = j;

        if (word[i] === first && i < word.length - 1 && word[i + 1] === second) {
          newWord.push(first + second);
          i += 2;
        } else {
          newWord.push(word[i]);
          i += 1;
        }
      }

      word = newWord;
      if (word.length === 1) break;
      pairs = getPairs(word);
    }

    const result = word.join(" ");
    this.cache.set(token, result);
    return result;
  }

  // ── encode ───────────────────────────────────────────────────────────────

  /**
   * Encode a string to a list of BPE token ids.
   * @param   {string}   text
   * @returns {number[]}
   */
  encode(text) {
    const bpeTokens = [];
    const utf8 = new TextEncoder();

    for (const match of text.matchAll(this.pat)) {
      const tokenStr = match[0];

      // Map each UTF-8 byte through byteEncoder
      const bytes      = utf8.encode(tokenStr);
      const mappedStr  = Array.from(bytes, (b) => this.byteEncoder[b]).join("");

      for (const bpePiece of this.bpe(mappedStr).split(" ")) {
        const id = this.encoder[bpePiece];
        if (id !== undefined) bpeTokens.push(id);
      }
    }
    return bpeTokens;
  }

  // ── decode ───────────────────────────────────────────────────────────────

  /**
   * Decode a list of BPE token ids back to a string.
   * @param   {number[]} tokens
   * @returns {string}
   */
  decode(tokens) {
    const chars = tokens.map((t) => this.decoder[t] ?? "").join("");
    const bytes = new Uint8Array([...chars].map((c) => this.byteDecoder[c]));
    return new TextDecoder("utf-8", { fatal: this.errors === "strict" }).decode(bytes);
  }
}

// ─── factory helpers ─────────────────────────────────────────────────────────

/**
 * Build an Encoder from raw file contents (strings).
 * Drop-in replacement for the Python get_encoder() when you already have
 * the file contents in memory (e.g. bundled, fetched, etc.).
 *
 * @param {string} encoderJson  - contents of encoder.json
 * @param {string} vocabBpe     - contents of vocab.bpe
 * @returns {Encoder}
 */
function getEncoder(encoderJson, vocabBpe) {
  const encoder = JSON.parse(encoderJson);
  const bpeMerges = vocabBpe
    .split("\n")
    .slice(1)                              // skip "#version: ..." header line
    .filter((line) => line.trim() !== "")  // drop trailing empty lines
    .map((line) => /** @type {[string,string]} */ (line.split(" ")));
  return new Encoder(encoder, bpeMerges);
}

/**
 * Async helper that mirrors the Python get_encoder(model_name, models_dir).
 * Reads encoder.json and vocab.bpe from `<modelsDir>/<modelName>/`.
 *
 * @param {string} modelName
 * @param {string} [modelsDir="."]
 * @returns {Promise<Encoder>}
 */
async function getEncoderFromFiles(modelName, modelsDir = ".") {
  const base        = join(modelsDir, modelName);
  const encoderJson = await readFile(join(base, "encoder.json"), "utf8");
  const vocabBpe    = await readFile(join(base, "vocab.bpe"),    "utf8");
  return getEncoder(encoderJson, vocabBpe);
}

// ─── internal util ───────────────────────────────────────────────────────────

/** Inclusive-start, exclusive-end integer range. */
function range(start, end) {
  return Array.from({ length: end - start }, (_, i) => start + i);
}

// ─── exports ─────────────────────────────────────────────────────────────────

export { Encoder, getEncoder, getEncoderFromFiles, bytesToUnicode, getPairs };
