/*Byte pair encoding utilities.

Copied from: https://github.com/openai/gpt-2/blob/master/src/encoder.py.
Transpiled into JS by Claude.
*/


let _bytesToUnicodeCache = null;
function bytesToUnicode() {
  if (_bytesToUnicodeCache) return _bytesToUnicodeCache;
 
  const bs = [
    ...range(ord("!"), ord("~") + 1),
    ...range(ord("¡"), ord("¬") + 1),
    ...range(ord("®"), ord("ÿ") + 1),
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
  const result = {};
  for (let i = 0; i < bs.length; i++) {
    result[bs[i]] = String.fromCodePoint(cs[i]);
  }
  _bytesToUnicodeCache = result;
  return result;
}
 
/**
 * Returns the set of symbol pairs in a word (represented as a tuple/array of symbols).
 */
function getPairs(word) {
  const pairs = new Set();
  let prevChar = word[0];
  for (let i = 1; i < word.length; i++) {
    pairs.add(`${prevChar}\x00${word[i]}`); // use null byte as delimiter
    prevChar = word[i];
  }
  return pairs;
}
 
/** Helper: range(start, end) — end exclusive */
function range(start, end) {
  return Array.from({ length: end - start }, (_, i) => start + i);
}
 
/** Helper: char code point */
function ord(char) {
  return char.codePointAt(0);
}
 
class Encoder {
  /**
   * @param {Object} encoder    - token string → token id
   * @param {Array}  bpeMerges - array of [a, b] merge pairs
   * @param {string} errors    - error handling mode (unused in JS, kept for parity)
   */
  constructor(encoder, bpeMerges, errors = "replace") {
    this.encoder = encoder;
    this.decoder = Object.fromEntries(
      Object.entries(encoder).map(([k, v]) => [v, k])
    );
    this.errors = errors;
    this.byteEncoder = bytesToUnicode();
    this.byteDecoder = Object.fromEntries(
      Object.entries(this.byteEncoder).map(([k, v]) => [v, Number(k)])
    );
    this.bpeRanks = new Map(bpeMerges.map((pair, i) => [pair.join("\x00"), i]));
    this.cache = new Map();
 
    // Matches contractions, words, numbers, punctuation, whitespace
    this.pat = new RegExp(
      `'s|'t|'re|'ve|'m|'ll|'d| ?[\\p{L}]+| ?[\\p{N}]+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+`,
      "gu"
    );
  }
 
  bpe(token) {
    if (this.cache.has(token)) return this.cache.get(token);
 
    let word = [...token]; // split into individual chars
    let pairs = getPairs(word);
 
    if (pairs.size === 0) return token;
 
    while (true) {
      // Find the lowest-rank bigram
      let bigram = null;
      let minRank = Infinity;
      for (const pair of pairs) {
        const rank = this.bpeRanks.has(pair) ? this.bpeRanks.get(pair) : Infinity;
        if (rank < minRank) {
          minRank = rank;
          bigram = pair;
        }
      }
 
      if (!this.bpeRanks.has(bigram)) break;
 
      const [first, second] = bigram.split("\x00");
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
 
  encode(text) {
    const bpeTokens = [];
    for (const token of text.matchAll(this.pat)) {
      const tokenStr = token[0];
      // Encode to UTF-8 bytes, then map each byte through byteEncoder
      const encoded = new TextEncoder().encode(tokenStr);
      const mappedToken = Array.from(encoded)
        .map((b) => this.byteEncoder[b])
        .join("");
      for (const bpeToken of this.bpe(mappedToken).split(" ")) {
        bpeTokens.push(this.encoder[bpeToken]);
      }
    }
    return bpeTokens;
  }
 
  decode(tokens) {
    const text = tokens.map((t) => this.decoder[t]).join("");
    const bytes = [...text].map((c) => this.byteDecoder[c]);
    return new TextDecoder("utf-8").decode(new Uint8Array(bytes));
  }
}
 
/**
 * Load an Encoder from encoder.json and vocab.bpe file contents.
 *
 * @param {string} encoderJson  - contents of encoder.json
 * @param {string} vocabBpe     - contents of vocab.bpe
 * @returns {Encoder}
 */
function getEncoder(encoderJson, vocabBpe) {
  const encoder = JSON.parse(encoderJson);
  const bpeMerges = vocabBpe
    .split("\n")
    .slice(1)                        // skip header line
    .filter((line) => line.trim())   // drop empty lines
    .map((line) => line.split(" ")); // ["a", "b"]
  return new Encoder({ encoder, bpeMerges });
}
 
// ─── Exports ────────────────────────────────────────────────────────────────
 
// CommonJS
if (typeof module !== "undefined" && module.exports) {
  module.exports = { Encoder, getEncoder, bytesToUnicode, getPairs };
}
 
// ESM
export { Encoder, getEncoder, bytesToUnicode, getPairs };
 