import express from "express";
import tokenizer from "./tokenizer.js";

const { encode, decode, tokenToId } = tokenizer;
const app = express();

app.use(express.json());

// 1. Health Check
app.get("/", (_req, res) => {
  res.json({ status: "ok" });
});

// 2. Vocabulary Metadata Snapshot
app.get("/vocab", (_req, res) => {
  const allVocabEntries = Object.entries(tokenToId);

  res.json({
    vocabSize: allVocabEntries.length,
    vocabSample: Object.fromEntries(allVocabEntries.slice(0, 50)),
  });
});

// 3. Text-to-Tokens Encoding (Human Text -> Machine IDs)
app.post("/encode", (req, res) => {
  const { text, addBos = false, addEos = false } = req.body || {};

  if (typeof text !== "string") {
    return res
      .status(400)
      .json({ error: "String payload 'text' is required." });
  }

  const tokenIds = encode(text, { addBos, addEos });
  res.json({ ids: tokenIds });
});

// 4. Tokens-to-Text Decoding (Machine IDs -> Human Text)
app.post("/decode", (req, res) => {
  const { ids, stripSpecial = true } = req.body || {};

  if (!Array.isArray(ids)) {
    return res
      .status(400)
      .json({ error: "Array of integers 'ids' is required." });
  }

  const decodedText = decode(ids, { stripSpecial });
  res.json({ text: decodedText });
});

const PORT = 3000;
app.listen(PORT, () => {
  console.log(`Tokenizer service running on http://localhost:${PORT}`);
});
