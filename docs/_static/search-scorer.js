var Scorer = {
  objNameMatch: 11,
  objPartialMatch: 6,
  objPrio: {
    0: 15,
    1: 5,
    2: -5,
  },
  objPrioDefault: 0,
  title: 15,
  partialTitle: 7,
  term: 5,
  partialTerm: 2,
  score: ([docName, _title, _anchor, _description, score, _filename, kind]) => {
    if (docName.startsWith("api-reference/")) return score - 20;
    if (kind === "title") return score + 5;
    // Partial object match should be demoted below text match
    if (kind === "object" && score == 5) return 4;
    // Title match in config index pages are not important
    if (/^systems\/[^/]+\/(?:train|eval)$/.test(docName) && score == 15) return 4;
    return score;
  },
};
