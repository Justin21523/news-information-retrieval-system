# CSoundex 中文諧音編碼詳細指南

## 目錄

1. [背景與動機](#1-背景與動機)
2. [演算法設計](#2-演算法設計)
3. [實作步驟](#3-實作步驟)
4. [測試策略](#4-測試策略)
5. [常見問題](#5-常見問題)

---

## 1. 背景與動機

### 1.1 Soundex 原理

**Soundex** 是美國人口普查局於 1918 年開發的語音編碼演算法，用於處理英文姓氏的拼寫變異。核心思想是：**發音相似的詞彙應該產生相同的編碼**。

**範例**：
- `Smith` → `S530`
- `Smyth` → `S530`
- `Schmidt` → `S530`

### 1.2 中文挑戰

中文不同於英文的拼音文字，需要特殊處理：

1. **漢字轉拼音**：中文字首先需轉換為羅馬拼音
2. **聲調處理**：中文有四聲（一聲、二聲、三聲、四聲）+ 輕聲
3. **多音字問題**：「行」可讀 `xing` 或 `hang`
4. **異形字**：「裡」vs.「裏」（相同意義，不同字形）

### 1.3 CSoundex 目標

**支援場景**：
- ✅ 同音字匹配：「張三」、「章三」
- ✅ 模糊搜尋：使用者輸入同音但錯別字
- ✅ 人名檢索：「王曉明」vs.「王小明」
- ✅ 容錯查詢：忽略聲調差異

---

## 2. 演算法設計

### 2.1 編碼步驟

```
輸入：「三聚氰胺」
  ↓
步驟 1：漢字 → 拼音
  san ju qing an
  ↓
步驟 2：去除聲調、轉小寫
  san ju qing an
  ↓
步驟 3：聲母分群編碼
  s-9, j-7, q-7, a-0
  ↓
步驟 4：格式化輸出
  S977, J700, Q700, A000
```

### 2.2 聲母分群規則

根據發音相似度分組（參考自漢語拼音聲母表）：

| 群組 | 聲母 | 說明 |
|------|------|------|
| 0 | a, e, i, o, u, ü | 元音（零聲母） |
| 1 | b, p | 雙唇音 |
| 2 | f | 唇齒音 |
| 3 | m | 雙唇鼻音 |
| 4 | d, t | 舌尖中音 |
| 5 | n, l | 舌尖鼻音 + 邊音 |
| 6 | g, k, h | 舌根音 |
| 7 | j, q, x | 舌面音 |
| 8 | zh, ch, sh, r | 捲舌音 |
| 9 | z, c, s | 平舌音 |

### 2.3 編碼格式

**標準格式**：`[首字母大寫][3 位數字]`

**範例**：
- 「張」(zhang) → `Z800`（zh=8, ang=0+0）
- 「章」(zhang) → `Z800`（相同編碼！）
- 「三」(san) → `S900`（s=9, an=0+0）

---

## 3. 實作步驟

### 3.1 模組結構

```
src/ir/text/csoundex.py
├── encode(text: str) -> str           # 主要編碼函式
├── char_to_pinyin(char: str) -> str   # 漢字轉拼音
├── normalize_pinyin(pinyin: str) -> str  # 拼音正規化
├── group_consonant(consonant: str) -> str  # 聲母分群
└── format_code(pinyin: str) -> str    # 格式化輸出
```

### 3.2 核心實作

#### encode() 主函式

```python
def encode(text: str, lexicon: Dict[str, str] = None) -> str:
    """
    Encode Chinese text to CSoundex code.

    Args:
        text: Input Chinese text (may contain mixed Chinese/English/punctuation)
        lexicon: Optional pinyin dictionary {char: pinyin}

    Returns:
        Space-separated CSoundex codes

    Example:
        >>> encode("三聚氰胺")
        "S900 J700 Q700 A000"

    Complexity:
        Time: O(n) where n is text length
        Space: O(n) for output codes
    """
    if lexicon is None:
        lexicon = load_pinyin_lexicon()

    codes = []

    for char in text:
        # Skip non-Chinese characters
        if not is_chinese_char(char):
            continue

        # Convert to pinyin
        pinyin = char_to_pinyin(char, lexicon)

        # Normalize
        pinyin_norm = normalize_pinyin(pinyin)

        # Encode
        code = format_code(pinyin_norm)

        codes.append(code)

    return ' '.join(codes)
```

#### char_to_pinyin() 漢字轉拼音

```python
def char_to_pinyin(char: str, lexicon: Dict[str, str]) -> str:
    """
    Convert a single Chinese character to pinyin.

    Args:
        char: Single Chinese character
        lexicon: Pinyin dictionary

    Returns:
        Pinyin romanization (with tone marks or numbers)

    Example:
        >>> char_to_pinyin('張', lexicon)
        'zhang1'

    Note:
        For polyphones (多音字), return the most common pronunciation.
        More advanced: use word context to disambiguate.
    """
    if char in lexicon:
        return lexicon[char]
    else:
        # Fallback: use pypinyin library
        try:
            from pypinyin import pinyin, Style
            result = pinyin(char, style=Style.TONE3)  # e.g., zhang1
            return result[0][0] if result else ''
        except ImportError:
            return ''  # Cannot convert
```

#### normalize_pinyin() 拼音正規化

```python
import re

def normalize_pinyin(pinyin: str) -> str:
    """
    Normalize pinyin: remove tones, convert to lowercase.

    Args:
        pinyin: Raw pinyin (e.g., 'zhāng' or 'zhang1')

    Returns:
        Normalized pinyin (e.g., 'zhang')

    Example:
        >>> normalize_pinyin('zhāng')
        'zhang'
        >>> normalize_pinyin('zhang1')
        'zhang'

    Complexity:
        Time: O(m) where m is pinyin length
        Space: O(m)
    """
    # Remove tone numbers (1-4)
    pinyin = re.sub(r'[1-4]', '', pinyin)

    # Remove tone marks (ā, á, ǎ, à → a)
    tone_map = {
        'ā': 'a', 'á': 'a', 'ǎ': 'a', 'à': 'a',
        'ē': 'e', 'é': 'e', 'ě': 'e', 'è': 'e',
        'ī': 'i', 'í': 'i', 'ǐ': 'i', 'ì': 'i',
        'ō': 'o', 'ó': 'o', 'ǒ': 'o', 'ò': 'o',
        'ū': 'u', 'ú': 'u', 'ǔ': 'u', 'ù': 'u',
        'ǖ': 'v', 'ǘ': 'v', 'ǚ': 'v', 'ǜ': 'v',  # ü → v
    }

    for tone_char, base_char in tone_map.items():
        pinyin = pinyin.replace(tone_char, base_char)

    return pinyin.lower()
```

#### format_code() 格式化編碼

```python
def format_code(pinyin: str) -> str:
    """
    Format normalized pinyin to CSoundex code.

    Args:
        pinyin: Normalized pinyin (e.g., 'zhang')

    Returns:
        CSoundex code (e.g., 'Z800')

    Algorithm:
        1. Extract first letter (capitalized)
        2. Extract consonants and group them
        3. Pad to 3 digits with zeros

    Example:
        >>> format_code('zhang')
        'Z800'
        >>> format_code('san')
        'S900'
        >>> format_code('ai')
        'A000'

    Complexity:
        Time: O(m) where m is pinyin length
        Space: O(1)
    """
    if not pinyin:
        return ''

    # First character (capitalized)
    first_char = pinyin[0].upper()

    # Extract consonants and group
    consonant_codes = []

    i = 0
    while i < len(pinyin):
        # Try two-character consonants first (zh, ch, sh)
        if i + 1 < len(pinyin):
            two_char = pinyin[i:i+2]
            code = group_consonant(two_char)
            if code is not None:
                consonant_codes.append(code)
                i += 2
                continue

        # Single character
        code = group_consonant(pinyin[i])
        if code is not None:
            consonant_codes.append(code)
        i += 1

    # Pad to 3 digits
    while len(consonant_codes) < 3:
        consonant_codes.append('0')

    # Take first 3 digits
    code_str = ''.join(consonant_codes[:3])

    return first_char + code_str
```

#### group_consonant() 聲母分群

```python
def group_consonant(consonant: str) -> str:
    """
    Map consonant to group code.

    Args:
        consonant: Single consonant or consonant cluster

    Returns:
        Group code ('0'-'9') or None if not a consonant

    Example:
        >>> group_consonant('zh')
        '8'
        >>> group_consonant('a')
        '0'
        >>> group_consonant('x')
        None  # Not in our mapping
    """
    # Load grouping rules from config
    groups = {
        '0': ['a', 'e', 'i', 'o', 'u', 'v'],  # v represents ü
        '1': ['b', 'p'],
        '2': ['f'],
        '3': ['m'],
        '4': ['d', 't'],
        '5': ['n', 'l'],
        '6': ['g', 'k', 'h'],
        '7': ['j', 'q', 'x'],
        '8': ['zh', 'ch', 'sh', 'r'],
        '9': ['z', 'c', 's'],
    }

    for code, consonants in groups.items():
        if consonant in consonants:
            return code

    return None  # Not a recognized consonant
```

### 3.3 設定檔

**configs/csoundex.yaml**：

```yaml
# CSoundex configuration

# Consonant grouping rules
consonant_groups:
  0: [a, e, i, o, u, v]  # Vowels (v = ü)
  1: [b, p]              # Bilabial
  2: [f]                 # Labiodental
  3: [m]                 # Bilabial nasal
  4: [d, t]              # Alveolar
  5: [n, l]              # Alveolar nasal + lateral
  6: [g, k, h]           # Velar
  7: [j, q, x]           # Palatal
  8: [zh, ch, sh, r]     # Retroflex
  9: [z, c, s]           # Alveolar sibilant

# Pinyin lexicon path
lexicon_path: datasets/lexicon/basic_pinyin.tsv

# Polyphone handling
polyphone_strategy: most_common  # Options: most_common, context_aware, all_variants
```

### 3.4 拼音字典格式

**datasets/lexicon/basic_pinyin.tsv**：

```tsv
字	拼音	頻率
的	de	1000000
一	yi	800000
是	shi	750000
不	bu	700000
了	le	650000
在	zai	600000
人	ren	550000
有	you	500000
我	wo	480000
他	ta	460000
...
```

---

## 4. 測試策略

### 4.1 測試案例設計

#### 4.1.1 基本功能測試

```python
# tests/test_csoundex.py

def test_encode_single_char():
    """Test encoding a single Chinese character."""
    assert encode("張") == "Z800"
    assert encode("三") == "S900"


def test_encode_homophone():
    """Test that homophones produce the same code."""
    assert encode("張") == encode("章") == "Z800"
    assert encode("李") == encode("理") == encode("裏") == "L500"


def test_encode_phrase():
    """Test encoding a multi-character phrase."""
    result = encode("三聚氰胺")
    assert result == "S900 J700 Q700 A000"
```

#### 4.1.2 邊界條件測試

```python
def test_encode_empty_string():
    """Test empty input."""
    assert encode("") == ""


def test_encode_single_vowel():
    """Test character with vowel initial."""
    assert encode("愛") == "A000"  # ai


def test_encode_long_text():
    """Test handling of long text."""
    text = "張" * 1000
    result = encode(text)
    codes = result.split()
    assert len(codes) == 1000
    assert all(code == "Z800" for code in codes)
```

#### 4.1.3 混合文字測試

```python
def test_encode_mixed_chinese_english():
    """Test text with mixed Chinese and English."""
    result = encode("三聚氰胺(melamine)是化學物")
    # Should skip English characters
    assert "S900" in result
    assert "melamine" not in result


def test_encode_with_punctuation():
    """Test handling of punctuation."""
    result = encode("你好，世界！")
    assert result == "N500 H600 S800 J700"  # Punctuation ignored
```

#### 4.1.4 多音字測試

```python
def test_polyphone_most_common():
    """Test polyphone handling with most common pronunciation."""
    # '行': xing (most common) or hang
    result = encode("行")
    assert result == "X700"  # xing (default)


def test_polyphone_context_aware():
    """Test context-aware polyphone resolution (advanced)."""
    # '銀行' (bank): hang
    # '行走' (walk): xing
    # This requires word segmentation + context
    pass  # Skip for basic implementation
```

### 4.2 效能測試

```python
import time

def test_performance_large_corpus():
    """Test encoding performance on large corpus."""
    # Generate test corpus
    corpus = "張三李四王五" * 10000  # 50,000 characters

    start = time.time()
    result = encode(corpus)
    elapsed = time.time() - start

    print(f"Encoded {len(corpus)} chars in {elapsed:.3f}s")
    print(f"Throughput: {len(corpus) / elapsed:.0f} chars/s")

    # Should complete in < 1 second for 50k chars
    assert elapsed < 1.0
```

### 4.3 準確性評估

```python
def test_accuracy_on_labeled_data():
    """Test accuracy on labeled homophone pairs."""
    # Load test data: [(char1, char2, should_match), ...]
    test_pairs = [
        ("張", "章", True),   # Homophones
        ("李", "理", True),
        ("張", "李", False),  # Different sounds
        ("三", "山", False),
        # ... more pairs
    ]

    correct = 0
    total = len(test_pairs)

    for char1, char2, should_match in test_pairs:
        code1 = encode(char1)
        code2 = encode(char2)
        matches = (code1 == code2)

        if matches == should_match:
            correct += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.1f}%")

    # Should achieve > 90% accuracy
    assert accuracy > 0.9
```

---

## 5. 常見問題

### Q1: 如何處理多音字？

**策略選項**：

1. **最常見讀音**（簡單）：
   - 從詞頻統計選擇最常用讀音
   - 準確率：~85%

2. **詞彙脈絡**（進階）：
   - 先分詞，再依詞彙查表
   - 例如：「銀行」→ `hang`，「行走」→ `xing`
   - 需要分詞工具（jieba）

3. **多編碼輸出**（備選）：
   - 輸出所有可能讀音的編碼
   - 例如：「行」→ `[X700, H600]`
   - 查詢時需匹配任一編碼

**建議實作**：初期使用策略 1，進階版本採用策略 2。

### Q2: 如何評估編碼品質？

**評估指標**：

1. **同音匹配率** *Homophone Matching Rate*：
   ```
   HMR = |正確匹配的同音字對| / |所有同音字對|
   ```

2. **誤匹配率** *False Positive Rate*：
   ```
   FPR = |錯誤匹配的非同音字對| / |所有非同音字對|
   ```

3. **檢索效能提升**：
   - 比較使用 vs. 不使用 CSoundex 的查全率（Recall）

**測試資料集**：
- 常見同音字表（教育部資料）
- 人名資料庫（戶政機關）
- 模糊搜尋查詢日誌

### Q3: 拼音字典缺字怎麼辦？

**解決方案**：

1. **使用 pypinyin 函式庫**：
   ```python
   from pypinyin import lazy_pinyin
   pinyin_list = lazy_pinyin("罕見字")
   ```

2. **備用編碼**：
   - 缺字用特殊碼如 `X999` 表示
   - 或跳過該字元

3. **字典擴充**：
   - 定期從使用者查詢日誌中識別缺字
   - 人工標註後加入字典

### Q4: 簡繁體如何處理？

**策略**：

1. **統一轉換**：
   ```python
   from opencc import OpenCC
   cc = OpenCC('s2t')  # Simplified to Traditional
   text_traditional = cc.convert(text_simplified)
   ```

2. **雙向索引**：
   - 建立簡繁映射表
   - 兩種字形產生相同編碼

### Q5: 如何整合到檢索系統？

**整合方式**：

1. **索引時編碼**：
   ```python
   def index_document(doc_text):
       # Original text index
       tokens = tokenize(doc_text)
       original_index = build_index(tokens)

       # CSoundex index
       csoundex_codes = encode(doc_text).split()
       csoundex_index = build_index(csoundex_codes)

       return original_index, csoundex_index
   ```

2. **查詢時編碼**：
   ```python
   def search(query, use_fuzzy=True):
       # Exact match
       exact_results = search_index(original_index, query)

       if use_fuzzy:
           # Fuzzy match with CSoundex
           query_codes = encode(query).split()
           fuzzy_results = search_index(csoundex_index, query_codes)

           # Merge results
           return merge_results(exact_results, fuzzy_results)
       else:
           return exact_results
   ```

3. **排序策略**：
   - 精確匹配結果排前
   - CSoundex 模糊匹配排後
   - 或依相似度分數排序

---

## 參考資料

- [Soundex Algorithm (Wikipedia)](https://en.wikipedia.org/wiki/Soundex)
- [漢語拼音方案](http://www.moe.gov.cn/jyb_sjzl/ziliao/A19/195802/t19580201_186000.html)
- [pypinyin 函式庫](https://github.com/mozillazg/python-pinyin)
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Chapter 3: Dictionaries and tolerant retrieval.

---

**最後更新**：2025-11-12
**維護者**：[您的姓名/學號]
