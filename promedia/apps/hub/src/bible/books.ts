/**
 * Protestant canon (66 books) — Spanish display names + English API slugs.
 * Chapter counts are standard Protestant versification.
 */

export type BibleBook = {
  id: string;
  nameEs: string;
  nameEn: string;
  /** Alternate names / abbreviations for reference search. */
  aliases: string[];
  chapters: number;
};

export const BIBLE_BOOKS: readonly BibleBook[] = [
  { id: "genesis", nameEs: "Génesis", nameEn: "Genesis", aliases: ["gen", "gn", "ge"], chapters: 50 },
  { id: "exodus", nameEs: "Éxodo", nameEn: "Exodus", aliases: ["exo", "ex", "exod"], chapters: 40 },
  { id: "leviticus", nameEs: "Levítico", nameEn: "Leviticus", aliases: ["lev", "lv", "le"], chapters: 27 },
  { id: "numbers", nameEs: "Números", nameEn: "Numbers", aliases: ["num", "nm", "nu"], chapters: 36 },
  { id: "deuteronomy", nameEs: "Deuteronomio", nameEn: "Deuteronomy", aliases: ["deut", "dt", "de"], chapters: 34 },
  { id: "joshua", nameEs: "Josué", nameEn: "Joshua", aliases: ["josh", "jos", "js"], chapters: 24 },
  { id: "judges", nameEs: "Jueces", nameEn: "Judges", aliases: ["judg", "jdg", "ju"], chapters: 21 },
  { id: "ruth", nameEs: "Rut", nameEn: "Ruth", aliases: ["ru", "rt"], chapters: 4 },
  {
    id: "1samuel",
    nameEs: "1 Samuel",
    nameEn: "1 Samuel",
    aliases: ["1sam", "1 sa", "1sm", "i samuel"],
    chapters: 31,
  },
  {
    id: "2samuel",
    nameEs: "2 Samuel",
    nameEn: "2 Samuel",
    aliases: ["2sam", "2 sa", "2sm", "ii samuel"],
    chapters: 24,
  },
  {
    id: "1kings",
    nameEs: "1 Reyes",
    nameEn: "1 Kings",
    aliases: ["1kgs", "1 reyes", "1ki", "i kings", "1 re"],
    chapters: 22,
  },
  {
    id: "2kings",
    nameEs: "2 Reyes",
    nameEn: "2 Kings",
    aliases: ["2kgs", "2 reyes", "2ki", "ii kings", "2 re"],
    chapters: 25,
  },
  {
    id: "1chronicles",
    nameEs: "1 Crónicas",
    nameEn: "1 Chronicles",
    aliases: ["1chr", "1 cr", "1cro", "i chronicles"],
    chapters: 29,
  },
  {
    id: "2chronicles",
    nameEs: "2 Crónicas",
    nameEn: "2 Chronicles",
    aliases: ["2chr", "2 cr", "2cro", "ii chronicles"],
    chapters: 36,
  },
  { id: "ezra", nameEs: "Esdras", nameEn: "Ezra", aliases: ["ezr", "esd"], chapters: 10 },
  { id: "nehemiah", nameEs: "Nehemías", nameEn: "Nehemiah", aliases: ["neh", "ne"], chapters: 13 },
  { id: "esther", nameEs: "Ester", nameEn: "Esther", aliases: ["est", "es"], chapters: 10 },
  { id: "job", nameEs: "Job", nameEn: "Job", aliases: ["jb"], chapters: 42 },
  { id: "psalms", nameEs: "Salmos", nameEn: "Psalms", aliases: ["ps", "psa", "salmo", "sal"], chapters: 150 },
  { id: "proverbs", nameEs: "Proverbios", nameEn: "Proverbs", aliases: ["prov", "prv", "pr"], chapters: 31 },
  { id: "ecclesiastes", nameEs: "Eclesiastés", nameEn: "Ecclesiastes", aliases: ["eccl", "ecc", "qohelet"], chapters: 12 },
  {
    id: "songofsolomon",
    nameEs: "Cantares",
    nameEn: "Song of Solomon",
    aliases: ["song", "sos", "cantares", "cant", "canto"],
    chapters: 8,
  },
  { id: "isaiah", nameEs: "Isaías", nameEn: "Isaiah", aliases: ["isa", "is"], chapters: 66 },
  { id: "jeremiah", nameEs: "Jeremías", nameEn: "Jeremiah", aliases: ["jer", "je"], chapters: 52 },
  { id: "lamentations", nameEs: "Lamentaciones", nameEn: "Lamentations", aliases: ["lam", "la"], chapters: 5 },
  { id: "ezekiel", nameEs: "Ezequiel", nameEn: "Ezekiel", aliases: ["ezek", "eze", "ez"], chapters: 48 },
  { id: "daniel", nameEs: "Daniel", nameEn: "Daniel", aliases: ["dan", "da", "dn"], chapters: 12 },
  { id: "hosea", nameEs: "Oseas", nameEn: "Hosea", aliases: ["hos", "os"], chapters: 14 },
  { id: "joel", nameEs: "Joel", nameEn: "Joel", aliases: ["jl", "joe"], chapters: 3 },
  { id: "amos", nameEs: "Amós", nameEn: "Amos", aliases: ["am"], chapters: 9 },
  { id: "obadiah", nameEs: "Abdías", nameEn: "Obadiah", aliases: ["obad", "ob", "abd"], chapters: 1 },
  { id: "jonah", nameEs: "Jonás", nameEn: "Jonah", aliases: ["jon", "jnh"], chapters: 4 },
  { id: "micah", nameEs: "Miqueas", nameEn: "Micah", aliases: ["mic", "mi", "mq"], chapters: 7 },
  { id: "nahum", nameEs: "Nahúm", nameEn: "Nahum", aliases: ["nah", "na"], chapters: 3 },
  { id: "habakkuk", nameEs: "Habacuc", nameEn: "Habakkuk", aliases: ["hab", "hb"], chapters: 3 },
  { id: "zephaniah", nameEs: "Sofonías", nameEn: "Zephaniah", aliases: ["zeph", "zep", "sof"], chapters: 3 },
  { id: "haggai", nameEs: "Hageo", nameEn: "Haggai", aliases: ["hag", "hg", "hag"], chapters: 2 },
  { id: "zechariah", nameEs: "Zacarías", nameEn: "Zechariah", aliases: ["zech", "zec", "zac"], chapters: 14 },
  { id: "malachi", nameEs: "Malaquías", nameEn: "Malachi", aliases: ["mal", "ml"], chapters: 4 },
  { id: "matthew", nameEs: "Mateo", nameEn: "Matthew", aliases: ["matt", "mt", "mat"], chapters: 28 },
  { id: "mark", nameEs: "Marcos", nameEn: "Mark", aliases: ["mk", "mr", "mar"], chapters: 16 },
  { id: "luke", nameEs: "Lucas", nameEn: "Luke", aliases: ["lk", "lc", "luc"], chapters: 24 },
  { id: "john", nameEs: "Juan", nameEn: "John", aliases: ["jn", "jhn", "jo"], chapters: 21 },
  { id: "acts", nameEs: "Hechos", nameEn: "Acts", aliases: ["act", "ac", "hech"], chapters: 28 },
  { id: "romans", nameEs: "Romanos", nameEn: "Romans", aliases: ["rom", "ro", "rm"], chapters: 16 },
  {
    id: "1corinthians",
    nameEs: "1 Corintios",
    nameEn: "1 Corinthians",
    aliases: ["1cor", "1 co", "1corintios", "i corinthians"],
    chapters: 16,
  },
  {
    id: "2corinthians",
    nameEs: "2 Corintios",
    nameEn: "2 Corinthians",
    aliases: ["2cor", "2 co", "2corintios", "ii corinthians"],
    chapters: 13,
  },
  { id: "galatians", nameEs: "Gálatas", nameEn: "Galatians", aliases: ["gal", "ga"], chapters: 6 },
  { id: "ephesians", nameEs: "Efesios", nameEn: "Ephesians", aliases: ["eph", "ef"], chapters: 6 },
  { id: "philippians", nameEs: "Filipenses", nameEn: "Philippians", aliases: ["phil", "php", "fil"], chapters: 4 },
  { id: "colossians", nameEs: "Colosenses", nameEn: "Colossians", aliases: ["col", "co"], chapters: 4 },
  {
    id: "1thessalonians",
    nameEs: "1 Tesalonicenses",
    nameEn: "1 Thessalonians",
    aliases: ["1thess", "1 ts", "1tes", "i thessalonians"],
    chapters: 5,
  },
  {
    id: "2thessalonians",
    nameEs: "2 Tesalonicenses",
    nameEn: "2 Thessalonians",
    aliases: ["2thess", "2 ts", "2tes", "ii thessalonians"],
    chapters: 3,
  },
  {
    id: "1timothy",
    nameEs: "1 Timoteo",
    nameEn: "1 Timothy",
    aliases: ["1tim", "1 ti", "1tm", "i timothy"],
    chapters: 6,
  },
  {
    id: "2timothy",
    nameEs: "2 Timoteo",
    nameEn: "2 Timothy",
    aliases: ["2tim", "2 ti", "2tm", "ii timothy"],
    chapters: 4,
  },
  { id: "titus", nameEs: "Tito", nameEn: "Titus", aliases: ["tit", "ti"], chapters: 3 },
  { id: "philemon", nameEs: "Filemón", nameEn: "Philemon", aliases: ["phlm", "phm", "flm"], chapters: 1 },
  { id: "hebrews", nameEs: "Hebreos", nameEn: "Hebrews", aliases: ["heb", "he"], chapters: 13 },
  { id: "james", nameEs: "Santiago", nameEn: "James", aliases: ["jas", "jm", "sant", "stgo"], chapters: 5 },
  {
    id: "1peter",
    nameEs: "1 Pedro",
    nameEn: "1 Peter",
    aliases: ["1pet", "1 pe", "1p", "i peter"],
    chapters: 5,
  },
  {
    id: "2peter",
    nameEs: "2 Pedro",
    nameEn: "2 Peter",
    aliases: ["2pet", "2 pe", "2p", "ii peter"],
    chapters: 3,
  },
  {
    id: "1john",
    nameEs: "1 Juan",
    nameEn: "1 John",
    aliases: ["1jn", "1 jo", "1jhn", "i john"],
    chapters: 5,
  },
  {
    id: "2john",
    nameEs: "2 Juan",
    nameEn: "2 John",
    aliases: ["2jn", "2 jo", "2jhn", "ii john"],
    chapters: 1,
  },
  {
    id: "3john",
    nameEs: "3 Juan",
    nameEn: "3 John",
    aliases: ["3jn", "3 jo", "3jhn", "iii john"],
    chapters: 1,
  },
  { id: "jude", nameEs: "Judas", nameEn: "Jude", aliases: ["jud", "jd"], chapters: 1 },
  {
    id: "revelation",
    nameEs: "Apocalipsis",
    nameEn: "Revelation",
    aliases: ["rev", "re", "apoc", "ap"],
    chapters: 22,
  },
];

function normalizeKey(s: string): string {
  return s
    .normalize("NFD")
    .replace(/\p{M}/gu, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .replace(/\s+/g, " ");
}

export function findBook(query: string): BibleBook | undefined {
  const q = normalizeKey(query);
  if (!q) {
    return undefined;
  }
  for (const book of BIBLE_BOOKS) {
    const candidates = [
      book.id,
      book.nameEs,
      book.nameEn,
      ...book.aliases,
    ].map(normalizeKey);
    if (candidates.includes(q)) {
      return book;
    }
  }
  // Prefix match (longer names first via scan order is fine for MVP)
  for (const book of BIBLE_BOOKS) {
    const candidates = [book.nameEs, book.nameEn, ...book.aliases].map(
      normalizeKey,
    );
    if (candidates.some((c) => c.startsWith(q) || q.startsWith(c))) {
      return book;
    }
  }
  return undefined;
}

export function getBookById(id: string): BibleBook | undefined {
  const key = id.trim().toLowerCase();
  return BIBLE_BOOKS.find((b) => b.id === key);
}
