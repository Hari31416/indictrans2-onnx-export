export const LANGUAGES = {
  eng_Latn: 'English',
  asm_Beng: 'Assamese (Bengali)',
  ben_Beng: 'Bengali',
  brx_Deva: 'Bodo',
  doi_Deva: 'Dogri',
  guj_Gujr: 'Gujarati',
  hin_Deva: 'Hindi',
  kan_Knda: 'Kannada',
  kas_Arab: 'Kashmiri',
  kok_Deva: 'Konkani',
  mai_Deva: 'Maithili',
  mal_Mlym: 'Malayalam',
  mni_Beng: 'Manipuri',
  mar_Deva: 'Marathi',
  npi_Deva: 'Nepali',
  ory_Orya: 'Odia',
  pan_Guru: 'Punjabi',
  san_Deva: 'Sanskrit',
  sat_Olck: 'Santali',
  snd_Arab: 'Sindhi',
  tam_Taml: 'Tamil',
  tel_Telu: 'Telugu',
  urd_Arab: 'Urdu',
}

const SCRIPT_RANGES = {
  pa: 0x0a00,
  gu: 0x0a80,
  or: 0x0b00,
  ta: 0x0b80,
  te: 0x0c00,
  kn: 0x0c80,
  ml: 0x0d00,
  si: 0x0d80,
  hi: 0x0900,
  mr: 0x0900,
  kK: 0x0900,
  sa: 0x0900,
  ne: 0x0900,
  sd: 0x0900,
  bn: 0x0980,
  as: 0x0980,
}

const FLORES_TO_ISO = {
  asm_Beng: 'as',
  ben_Beng: 'bn',
  brx_Deva: 'hi',
  doi_Deva: 'hi',
  guj_Gujr: 'gu',
  hin_Deva: 'hi',
  kan_Knda: 'kn',
  kas_Arab: 'ur',
  kok_Deva: 'hi',
  gom_Deva: 'kK',
  mai_Deva: 'hi',
  mal_Mlym: 'ml',
  mar_Deva: 'hi',
  mni_Beng: 'bn',
  npi_Deva: 'hi',
  ory_Orya: 'or',
  pan_Guru: 'pa',
  san_Deva: 'hi',
  sat_Olck: 'or',
  snd_Arab: 'ur',
  tam_Taml: 'ta',
  tel_Telu: 'te',
  urd_Arab: 'ur',
}

function correctTamilMapping(offset) {
  if (offset >= 0x15 && offset <= 0x28 && offset !== 0x1c) {
    const rem = (offset - 0x15) % 5
    if (rem !== 0 && rem !== 4) {
      const substChar = Math.floor((offset - 0x15) / 5)
      offset = 0x15 + 5 * substChar
    }
  }
  if (offset === 0x2b || offset === 0x2c || offset === 0x2d) {
    offset = 0x2a
  }
  if (offset === 0x36) {
    offset = 0x37
  }
  return offset
}

export function transliterate(text, srcLang, tgtLang) {
  const srcIso = FLORES_TO_ISO[srcLang]
  const tgtIso = FLORES_TO_ISO[tgtLang]

  if (!srcIso || !tgtIso) return text

  const srcStart = SCRIPT_RANGES[srcIso]
  const tgtStart = SCRIPT_RANGES[tgtIso]

  if (srcStart === undefined || tgtStart === undefined) return text

  const chars = []
  for (let i = 0; i < text.length; i++) {
    const c = text[i]
    const code = c.charCodeAt(0)
    let offset = code - srcStart

    if (offset >= 0 && offset <= 0x6f && c !== '\u0964' && c !== '\u0965') {
      if (tgtIso === 'ta') {
        offset = correctTamilMapping(offset)
      }
      chars.push(String.fromCharCode(tgtStart + offset))
    } else {
      chars.push(c)
    }
  }
  return chars.join('')
}
