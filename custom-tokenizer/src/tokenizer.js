import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// 1. Load and parse Vocabulary configuration
const vocabPath = join(__dirname, "..", "data", "vocab.json");
const rawVocabData = readFileSync(vocabPath, "utf8");
const vocab = JSON.parse(rawVocabData);

const tokenToId = vocab.tokenToId || {};
const idToToken = vocab.idToToken || {};
const specialTokens = vocab.specialTokens || {};

// 2. Extract special token control IDs
const PAD_ID = specialTokens["<PAD>"];
const UNK_ID = specialTokens["<UNK>"];
const BOS_ID = specialTokens["<BOS>"];
const EOS_ID = specialTokens["<EOS>"];

/**
 * Normalizes input text using standard Unicode decomposition/composition (NFKC)
 * to ensure consistency before tokenization processing.
 */
function normalizeText(text) {
  if (typeof text !== "string") return "";
  return text.normalize("NFKC");
}

/**
 * Encodes human-readable text into an array of numerical token IDs.
 */
function encode(text, options = {}) {
  const { addBos = false, addEos = false } = options;
  const normalizedText = normalizeText(text);
  const internalTokenIds = [];

  if (addBos && BOS_ID !== undefined) {
    internalTokenIds.push(BOS_ID);
  }

  // Character-level mapping loop
  for (const character of normalizedText) {
    const lowercaseChar = character.toLowerCase();

    if (Object.prototype.hasOwnProperty.call(tokenToId, lowercaseChar)) {
      internalTokenIds.push(tokenToId[lowercaseChar]);
    } else {
      internalTokenIds.push(UNK_ID);
    }
  }

  if (addEos && EOS_ID !== undefined) {
    internalTokenIds.push(EOS_ID);
  }

  return internalTokenIds;
}

/**
 * Decodes an array of numerical token IDs back into standard text strings.
 */
function decode(ids, options = {}) {
  const { stripSpecial = true } = options;

  if (!Array.isArray(ids)) return "";

  // Map individual IDs back to character tokens
  const mappedTokens = ids.map((id) => {
    const stringIdKey = String(id);
    return Object.prototype.hasOwnProperty.call(idToToken, stringIdKey)
      ? idToToken[stringIdKey]
      : "<UNK>";
  });

  let reconstructedText = mappedTokens.join("");

  if (stripSpecial) {
    reconstructedText = reconstructedText.replace(
      /<BOS>|<EOS>|<PAD>|<UNK>/g,
      "",
    );
  }

  return reconstructedText;
}

export default {
  encode,
  decode,
  tokenToId,
  idToToken,
  PAD_ID,
  UNK_ID,
  BOS_ID,
  EOS_ID,
};
