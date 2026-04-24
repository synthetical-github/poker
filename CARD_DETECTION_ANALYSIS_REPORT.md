# Poker Card Detection System - Comprehensive Analysis Report

**Generated:** April 23, 2026  
**Analysis Scope:** Complete card detection pipeline, templates, configuration, and image processing

---

## CRITICAL ISSUES

### 1. **Threshold Values Too Restrictive for Board Card Detection** (CRITICAL)
**Files:** `detectors/card_detector.py`

#### Issue 1A: Board Card Template Matching Threshold Too High
- **Location:** [detectors/card_detector.py](detectors/card_detector.py#L2044-L2052)
- **Severity:** CRITICAL
- **Description:** Board card detection requires `board_score >= 0.78 and (board_score - board_second) >= 0.02`, which is extremely high and will cause false negatives
- **Code:**
  ```python
  # Line 2044-2052: Board card detection
  if (board_score >= 0.98 and (board_score - board_second) >= 0.005)
      or (board_score >= 0.78 and (board_score - board_second) >= 0.02)
  ```
- **Problem:** The gap requirement of 0.02 between best and second-best match is too strict for real-world card variations
- **Impact:** Many valid board cards will fail to match, especially on Turn/River cards with slight image variations
- **Recommendation:** Reduce to `board_score >= 0.65 and gap >= 0.01` for better detection

#### Issue 1B: Card Match Threshold Set to 0.82
- **Location:** [detectors/card_detector.py](detectors/card_detector.py#L49-L51)
- **Severity:** CRITICAL
- **Description:** `self.min_card_match_threshold = 0.82` is too high for template matching
- **Code:**
  ```python
  self.min_card_match_threshold = 0.82
  self.min_card_match_gap = 0.03
  ```
- **Problem:** This threshold ignores valid card matches that score 0.75-0.81
- **Impact:** Regular cards fail to detect when templates have minor quality variations
- **Recommendation:** Reduce to 0.70-0.75 based on template quality analysis

#### Issue 1C: Minimum Gap Between Card Matches Too Large
- **Location:** [detectors/card_detector.py](detectors/card_detector.py#L51)
- **Problem:** `min_card_match_gap = 0.03` means the best match must be 3% better than second-best, which is unrealistic
- **Impact:** Similar-looking cards (e.g., 5H vs 5D) fail when templates are similar
- **Recommendation:** Use 0.015 or dynamically adjust based on top match confidence

### 2. **Rank and Suit Detection Thresholds Inconsistent** (CRITICAL)
**Files:** `detectors/card_detector.py`

#### Issue 2A: Corner Rank Threshold Too Low
- **Location:** [detectors/card_detector.py](detectors/card_detector.py#L53)
- **Severity:** CRITICAL
- **Code:**
  ```python
  self.min_corner_rank_match_threshold = 0.42
  ```
- **Problem:** Using 0.42 threshold for rank matching allows false positives (0->Q, 1->T, L->T misidentifications)
- **Impact:** Wrong rank detection (e.g., "2" detected as "Z", "5" as "S")
- **Recommendation:** Increase to 0.58-0.62 minimum

#### Issue 2B: Corner Suit Threshold Too Low
- **Location:** [detectors/card_detector.py](detectors/card_detector.py#L54)
- **Code:**
  ```python
  self.min_corner_suit_match_threshold = 0.36
  ```
- **Problem:** 0.36 is dangerously low - suits are only 4 symbols, so false matches are highly probable
- **Impact:** Heart detected as Diamond, Spade as Club, especially with low-quality images
- **Recommendation:** Increase to 0.50-0.58 minimum

### 3. **OCR Configuration Missing from Card Detection** (CRITICAL)
**Files:** `detectors/card_detector.py`

#### Issue 3A: No Tesseract Configuration in Card Detector
- **Location:** [detectors/card_detector.py](detectors/card_detector.py#L1-L20) (import section)
- **Severity:** CRITICAL
- **Problem:** Card detector imports pytesseract but never sets `pytesseract.pytesseract.tesseract_cmd`
- **Impact:** OCR fallback will fail on Windows if Tesseract is not in PATH
- **Code Fix Needed:**
  ```python
  # Missing in __init__:
  if os.name == 'nt':
      tesseract_cmd = LIVE_CONFIG.get('tesseract_cmd')
      if tesseract_cmd:
          pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
  ```
- **Recommendation:** Add Tesseract initialization to CardDetector.__init__()

#### Issue 3B: OCR Fallback Too Permissive
- **Location:** [detectors/card_detector.py](detectors/card_detector.py#L2088-L2095)
- **Problem:** OCR fallback accepts ANY detected card string without validation
- **Code:**
  ```python
  for rank in self.ranks:
       for suit in self.suits:
           if card_str in cleaned_text:  # Too permissive!
  ```
- **Impact:** If OCR detects "K5H A2D", it might match "K" with "Hearts" incorrectly
- **Recommendation:** Use proper card string parsing with position-based matching

### 4. **Template Loading Issues** (CRITICAL)
**Files:** `detectors/card_detector.py`, `config.py`

#### Issue 4A: Mixed Template Versions Not Checked for Completeness
- **Location:** [detectors/card_detector.py](detectors/card_detector.py#L35-L48)
- **Severity:** CRITICAL
- **Problem:** Code loads multiple template versions (v2, v3) but doesn't verify coverage
- **Code:**
  ```python
  self.board_rank_templates = self._load_merged_symbol_templates(
      ["board_rank_templates_v2", "board_rank_templates_v3"],
      valid_labels=set(self.ranks),
  )
  ```
- **Impact:** If v2 is incomplete and v3 only partially fills gaps, some ranks won't load
- **Recommendation:** 
  1. Log which ranks are missing after merge
  2. Add warnings if any rank/suit is completely missing
  3. Verify all 13 ranks and 4 suits are present

#### Issue 4B: No Validation of Card Template Completeness
- **Location:** [detectors/card_detector.py](detectors/card_detector.py#L103)
- **Problem:** `_load_card_templates()` doesn't verify all 52 cards are loaded
- **Expected:** 52 card templates (13 ranks × 4 suits)
- **Impact:** Missing cards will have no templates, causing detection failures
- **Recommendation:** Add validation:
  ```python
  expected_cards = {f"{r}{s}" for r in self.ranks for s in self.suits}
  loaded_cards = set(templates.keys())
  missing = expected_cards - loaded_cards
  if missing:
      logger.error(f"Missing card templates: {missing}")
  ```

### 5. **Table Layout Auto-Detection Unreliable** (CRITICAL)
**Files:** `detectors/card_detector.py`

#### Issue 5A: Layout Hint Detection Depends on OCR Without Fallback
- **Location:** [detectors/card_detector.py](detectors/card_detector.py#L1080-L1100)
- **Severity:** CRITICAL
- **Code:**
  ```python
  def _infer_layout_hint_from_title(self, screenshot_bgr: np.ndarray) -> Optional[str]:
      # ... OCR on window title area ...
      try:
          text = pytesseract.image_to_string(thresh, config='--psm 6')
      except Exception:
          return None  # Complete failure if OCR fails!
  ```
- **Problem:** If OCR fails, returns None and layout detection falls back to config
- **Impact:** Wrong table layout selected, all ROIs are off, all cards fail to detect
- **Recommendation:** Add config-based fallback, validate OCR result

#### Issue 5B: Primary/Candidate Layout Selection Logic Complex
- **Location:** [detectors/card_detector.py](detectors/card_detector.py#L1268-L1340)
- **Problem:** 10+ complex conditions for choosing between layouts - scores can be arbitrary
- **Impact:** Layout switching causes cascading detection failures
- **Recommendation:** Simplify with 3 clear rules: (1) 2 hole + 0-5 board → use primary, (2) missing cards with >2 gap → try backup, (3) backup score 14+ points better → switch

---

## HIGH SEVERITY ISSUES

### 6. **ROI Region Calculations Misaligned with Actual Card Positions** (HIGH)
**Files:** `config.py`, `detectors/card_detector.py`

#### Issue 6A: Hole Card Regions for acipayam_heads_up Layout
- **Location:** [config.py](config.py#L149-L156)
- **Severity:** HIGH
- **Code:**
  ```python
  'acipayam_heads_up': {
      'hero_hole_cards': [
          (836, 902, 92, 136),    # x, y, w, h
          (944, 902, 92, 136),
      ],
  ```
- **Problem:** Reference dimensions are (1935, 1369) but card regions may not match actual render at different resolutions
- **Impact:** Card detection searches wrong screen areas, missing cards completely
- **Verification Needed:** Run [calibrate_capture.py](calibrate_capture.py) to validate

#### Issue 6B: Board Card Spacing Assumptions
- **Location:** [config.py](config.py#L157-L167)
- **Problem:** Board cards assume fixed spacing between all 5 cards, but spacing changes with different layouts
- **Code:**
  ```python
  'community_cards': [
      (646, 492, 108, 160),  # Flop 1
      (764, 492, 108, 160),  # Flop 2
      (882, 492, 108, 160),  # Flop 3
      (1016, 492, 104, 160), # Turn - DIFFERENT WIDTH!
      (1138, 492, 104, 160), # River - DIFFERENT WIDTH!
  ```
- **Impact:** Turn/River cards have different ROI sizes, causing extraction failures
- **Recommendation:** Use consistent widths or apply dynamic scaling

#### Issue 6C: Scaling Function Precision Loss
- **Location:** [detectors/card_detector.py](detectors/card_detector.py#L2240-L2260)
- **Problem:** Integer division in scaling loses precision
- **Code:**
  ```python
  scaled_x = table_x + int((x / self.reference_width) * table_w)
  # At 1920x1080 vs 1935x1369 reference:
  # 646 / 1935 * 1920 = 644.9... → 644 (lost ~1px accuracy)
  ```
- **Impact:** ROI shifts by 1-3 pixels per card, compounding across all positions
- **Recommendation:** Use rounding instead of truncation, or pre-test on multiple resolutions

### 7. **Image Surface Extraction Has Hardcoded Thresholds** (HIGH)
**Files:** `detectors/card_detector.py`

#### Issue 7A: Card Surface Detection White Ratio Threshold
- **Location:** [detectors/card_detector.py](detectors/card_detector.py#L1525-L1530)
- **Severity:** HIGH
- **Code:**
  ```python
  white_ratio = cv2.countNonZero(white_mask) / float(white_mask.size)
  if white_ratio < 0.08:  # HARDCODED
      return default_inset_crop()
  ```
- **Problem:** 0.08 threshold is arbitrary and breaks on slightly yellowed/aged card images
- **Impact:** Cards with slightly discolored borders fail surface extraction
- **Recommendation:** Make configurable with 0.05-0.12 range

#### Issue 7B: Community Card Surface Detection Has Strict Aspect Ratio Check
- **Location:** [detectors/card_detector.py](detectors/card_detector.py#L1612-1616)
- **Code:**
  ```python
  if not 0.42 <= aspect_ratio <= 0.95:
      return False
  ```
- **Problem:** Aspect ratios 0.40-0.42 or 0.95-1.0 are rejected (e.g., slightly rotated cards)
- **Impact:** Slightly scaled/rotated cards fail detection
- **Recommendation:** Expand range to 0.38-1.05 with additional contour area checks

### 8. **Missing Boundary Checks in ROI Extraction** (HIGH)
**Files:** `detectors/card_detector.py`, `bot_logic.py`

#### Issue 8A: Image Boundary Violations in _extract_symbol_patch
- **Location:** [detectors/card_detector.py](detectors/card_detector.py#L344-371)
- **Severity:** HIGH
- **Problem:** While there are boundary checks, the logic with multiple nested crops could exceed bounds
- **Code:**
  ```python
  search = image[y1:y2, x1:x2]  # After fractional box extraction
  # Later:
  min_y = max(0, min(b[1] for b in include) - 2)  # -2 could underflow
  ```
- **Impact:** Rare edge cases cause crashes when card is at image edge
- **Recommendation:** Add final boundary validation after all crops

#### Issue 8B: bot_logic.py Has No Bounds Checking for Card Coordinates
- **Location:** [bot_logic.py](bot_logic.py#L94-117)
- **Code:**
  ```python
  community_card_y_threshold = self.current_table_coords[1] + int(self.current_table_coords[3] * 0.5)
  # No validation that current_table_coords is valid
  for card_name, loc in detected_cards_with_loc:
      if loc[1] < community_card_y_threshold:  # loc could be None/malformed
  ```
- **Impact:** Malformed card locations crash the game state update
- **Recommendation:** Add validation:
  ```python
  if loc is None or len(loc) < 4:
      continue
  if not (0 <= loc[0] < screenshot.shape[1]):
      continue  # Invalid x coordinate
  ```

### 9. **Detection Cascade Fails Silently** (HIGH)
**Files:** `detectors/card_detector.py`

#### Issue 9A: _process_card_roi Returns None Without Clear Logging
- **Location:** [detectors/card_detector.py](detectors/card_detector.py#L2063-2112)
- **Severity:** HIGH
- **Problem:** Multiple fallback mechanisms (template, corner, compact focus, OCR) but None logged when all fail
- **Code:**
  ```python
  if self.layout_name == 'acipayam_heads_up':
      self._store_cached_roi_result(cache_key, None)
      return None  # Silent failure - no log!
  ```
- **Impact:** Undetectable cards blend into "failed detection" with no diagnostics
- **Recommendation:** Add specific logging:
  ```python
  logger.debug(f"Card detection failed for {context}: "
      f"template_score={best_match_confidence:.2f}, "
      f"corner_card={corner_card}, "
      f"ocr_card={ocr_card}")
  ```

#### Issue 9B: Surface Extraction Failures Not Logged
- **Location:** [detectors/card_detector.py](detectors/card_detector.py#L1485-1497)
- **Problem:** If card surface extraction returns None, no indication why
- **Impact:** Can't distinguish between (card not on screen) vs (extraction failed)
- **Recommendation:** Log why surface extraction failed (no white border? bad aspect ratio? etc.)

### 10. **Layout-Specific Detection Logic Buried in Generic Code** (HIGH)
**Files:** `detectors/card_detector.py`

#### Issue 10A: acipayam_heads_up Special Cases Scattered
- **Location:** [detectors/card_detector.py](detectors/card_detector.py#L1173-1177), [Line 2046-2053], [Line 2082-2109]
- **Problem:** Multiple `if self.layout_name == "acipayam_heads_up"` checks scattered across file
- **Code Example:**
  ```python
  if self.layout_name == "acipayam_heads_up":
      return cropped  # Line 1547
  # ... 50 lines later ...
  if self.layout_name == 'acipayam_heads_up' and card_surface...:
      # Line 2046 different logic
  ```
- **Impact:** Hard to modify detection for one layout without breaking others, testing difficult
- **Recommendation:** Create separate `_process_card_roi_acipayam()` and `_process_card_roi_generic()` methods

#### Issue 10B: Compact Focus Detection Coupled to Headers-Up
- **Location:** [detectors/card_detector.py](detectors/card_detector.py#L910-970)
- **Problem:** `_detect_card_from_compact_focus()` is hardcoded for hero1/hero2/board contexts, inflexible
- **Impact:** Adding new layout requires rewriting this method
- **Recommendation:** Make focus boxes configurable per layout in config.py

### 11. **HSV Color Space Thresholds Not Calibrated for Lighting** (HIGH)
**Files:** `detectors/card_detector.py`

#### Issue 11A: Hardcoded Green/Blue/Red Hue Ranges
- **Location:** [detectors/card_detector.py](detectors/card_detector.py#L887-903)
- **Code:**
  ```python
  green_count = int(np.sum((hue >= 40) & (hue <= 95) & (sat > 50) & (val > 40)))
  blue_count = int(np.sum((hue >= 90) & (hue <= 135) & (sat > 50) & (val > 40)))
  red_count = int(np.sum((((hue <= 12) | (hue >= 170)) & (sat > 50) & (val > 40))))
  ```
- **Problem:** These ranges are specific to one lighting condition and will fail in:
  - Bright sunlight (saturation >> 80)
  - Dark table (value < 40)
  - Different card stock color profile
- **Impact:** Suit detection fails in different lighting conditions
- **Recommendation:** Calibrate ranges per layout/lighting or use adaptive thresholding

#### Issue 11B: Black Suit Detection Assumes Pixel Value < 70
- **Location:** [detectors/card_detector.py](detectors/card_detector.py#L1903)
- **Code:**
  ```python
  if gray_val < 70 or val < 70:
      black_count += 1
  ```
- **Problem:** Different screens/lighting may require 60 or 85 threshold
- **Impact:** Spades detected as Hearts/Diamonds in different conditions
- **Recommendation:** Make configurable, add calibration helper

### 12. **Cache Implementation Has No TTL/Invalidation** (HIGH)
**Files:** `detectors/card_detector.py`

#### Issue 12A: ROI Cache Never Expires
- **Location:** [detectors/card_detector.py](detectors/card_detector.py#L1000-1020)
- **Severity:** HIGH
- **Problem:** LRU cache keeps old results forever if layout changes mid-session
- **Code:**
  ```python
  def _clear_roi_cache(self) -> None:
      self._roi_result_cache.clear()  # Only cleared when refresh_layout called
  ```
- **Impact:** Changing table layout mid-session causes stale cached results to be used
- **Recommendation:** Add timestamp-based TTL or invalidate on layout change

#### Issue 12B: Frame Detection Cache Doesn't Handle Table Shift
- **Location:** [detectors/card_detector.py](detectors/card_detector.py#L1026-1042)
- **Problem:** Cache key is based on screenshot hash, not table position
- **Impact:** If window moves 10 pixels, old cached result still used
- **Recommendation:** Include table_coords in frame cache key

---

## MEDIUM SEVERITY ISSUES

### 13. **Text Parsing Fragile for Different Number Formats** (MEDIUM)
**Files:** `bot_logic.py`, `detectors/table_parser.py`, `detectors/card_detector.py`

#### Issue 13A: _parse_amount Assumes Specific Decimal Formats
- **Location:** [bot_logic.py](bot_logic.py#L172-190)
- **Severity:** MEDIUM
- **Code:**
  ```python
  if '.' in cleaned and ',' in cleaned:  # Assumes German format 1.234,56
      parts = cleaned.split(',')
      integer_part = parts[0].replace('.', '')
  elif '.' in cleaned and cleaned.count('.') > 1:  # Multiple dots = thousands sep
      cleaned = cleaned.replace('.', '')
  ```
- **Problem:** Fails on English format "1,234.56" or mixed formats
- **Impact:** Pot size and bet amounts misread
- **Recommendation:** Use regex: `r'[\d,.]+'` and try both formats

#### Issue 13B: OCR Character Whitelist Incomplete
- **Location:** [detectors/card_detector.py](detectors/card_detector.py#L820-821)
- **Code:**
  ```python
  config='--psm 10 -c tessedit_char_whitelist=A23456789TJQK10'
  ```
- **Problem:** Whitelist "10" but code expects 'T', causing parse failures
- **Recommendation:** Use only 'A23456789TJQK' and handle conversions in code

### 14. **No Validation of Detected Card Sequences** (MEDIUM)
**Files:** `bot_logic.py`, `detectors/card_detector.py`

#### Issue 14A: Community Cards Can Be Out of Order
- **Location:** [detectors/card_detector.py](detectors/card_detector.py#L1165-1180)
- **Problem:** Code sorts detected cards by X position but doesn't validate sequence:
  - Can't have 4 community cards (only 0, 3, 4, 5 valid)
  - Can't have duplicate cards (52 unique cards max)
- **Code:**
  ```python
  if len(best["community_cards"]) == 0 and primary_board_count == 0:  # Valid check exists
  # But no check: community_cards must not have duplicates
  ```
- **Impact:** Impossible game states accepted (e.g., 2♣, 2♣, 3♦, etc.)
- **Recommendation:** Add validation:
  ```python
  def _validate_game_state(hole_cards, community_cards):
      all_cards = hole_cards + community_cards
      if len(all_cards) != len(set(all_cards)):
          logger.error(f"Duplicate cards detected: {all_cards}")
          return False
      return True
  ```

#### Issue 14B: No Validation That Hole Cards ≠ Community Cards
- **Location:** [bot_logic.py](bot_logic.py#L117-119)
- **Problem:** Same card could be detected as both hole card and community card
- **Impact:** Invalid game state causes wrong decisions
- **Recommendation:** Validate disjoint sets:
  ```python
  if any(hc in community_cards for hc in hole_cards):
      logger.error("Hole card appears in community!")
  ```

### 15. **Error Recovery Mechanisms Missing** (MEDIUM)
**Files:** All detection files

#### Issue 15A: No Retry Logic for Transient Failures
- **Location:** [detectors/card_detector.py](detectors/card_detector.py#L1060-1070) (all detection methods)
- **Problem:** Single failure = False negative, no retry with slightly different parameters
- **Impact:** Flickering/noise in one frame loses all card data
- **Recommendation:** Implement 2-3 retry passes with:
  - Pass 1: Standard thresholds
  - Pass 2: ±10% threshold adjustment
  - Pass 3: Alternative template set (v2 vs v3)

#### Issue 15B: No Graceful Degradation
- **Location:** Entire pipeline
- **Problem:** Missing one card causes entire game state to be rejected
- **Code:** [bot_logic.py](bot_logic.py#L95-155) returns None on any error
- **Recommendation:** Return best-effort results with confidence scores:
  ```python
  game_state = {
      'hole_cards': [...],
      'community_cards': [...],
      'hole_cards_confidence': 0.95,  # NEW
      'community_confidence': 0.45,   # NEW - LOW, action shouldn't depend on board
      'uncertainty_flags': ['missing_turn_card'],
  }
  ```

### 16. **Symbol Template Loading Has No Verification** (MEDIUM)
**Files:** `detectors/card_detector.py`

#### Issue 16A: Merged Rank/Suit Templates Not Validated
- **Location:** [detectors/card_detector.py](detectors/card_detector.py#L166-193)
- **Problem:** Code loads rank/suit templates but doesn't verify all ranks/suits present
- **Code:**
  ```python
  self.hero_rank_templates = self._load_merged_symbol_templates(
      ["hero_rank_templates", "hero_rank_templates_v2", "hero_rank_templates_v3"],
      valid_labels=set(self.ranks),
  )
  # No check: Are all 13 ranks actually loaded?
  ```
- **Impact:** Missing ranks like 'T' or '9' silently cause detection failures
- **Recommendation:** Add post-load validation:
  ```python
  missing_ranks = set(self.ranks) - set(self.hero_rank_templates.keys())
  if missing_ranks:
      logger.error(f"Missing hero rank templates: {missing_ranks}")
  ```

#### Issue 16B: Corner Rank/Suit Templates Extracted Without Validation
- **Location:** [detectors/card_detector.py](detectors/card_detector.py#L416-445)
- **Problem:** Extracting corner symbols from full card templates, but no check if extraction succeeded
- **Code:**
  ```python
  if rank_roi is not None:
      normalized_rank = self._normalize_rank_image(rank_roi)
      if normalized_rank is not None and normalized_rank.size > 0:
          rank_templates.setdefault(parsed.rank, []).append(normalized_rank)
  # If all extractions fail for a rank: no templates for that rank
  ```
- **Impact:** Layout-specific detection fails if corner extraction is poor
- **Recommendation:** Log which ranks/suits failed extraction

### 17. **Tesseract OCR Not Integrated with Card Detection** (MEDIUM)
**Files:** `detectors/card_detector.py`

#### Issue 17A: OCR Only Used as Last Resort
- **Location:** [detectors/card_detector.py](detectors/card_detector.py#L2080-2112)
- **Severity:** MEDIUM
- **Problem:** OCR fallback skipped for acipayam_heads_up layout
- **Code:**
  ```python
  if self.layout_name == 'acipayam_heads_up':
      self._store_cached_roi_result(cache_key, None)
      return None  # OCR never attempted!
  ```
- **Impact:** If all template-based methods fail, just returns None instead of trying OCR
- **Recommendation:** Always try OCR if other methods fail, regardless of layout

#### Issue 17B: OCR Config Doesn't Match Card Types
- **Location:** [detectors/card_detector.py](detectors/card_detector.py#L820)
- **Code:**
  ```python
  config = '--psm 10 -c tessedit_char_whitelist=A23456789TJQK10'
  ```
- **Problem:** PSM 10 (single character) fails for 2-character card codes
- **Recommendation:** Use PSM 7 or 8 for single line of text

---

## LOW SEVERITY ISSUES

### 18. **Logging Could Be More Diagnostic** (LOW)
**Files:** All detection files

#### Issue 18A: Missing Debug Logs for Threshold Decisions
- **Location:** [detectors/card_detector.py](detectors/card_detector.py#L1260-1340)
- **Problem:** Complex layout selection logic has no logging of why layout was chosen
- **Recommendation:** Add:
  ```python
  logger.debug(f"Layout selection: primary={primary['score']:.1f}, "
      f"hole_cards={len(primary['hole_cards'])}, "
      f"board_cards={len(primary['community_cards'])}")
  ```

#### Issue 18B: Confidence Scores Not Exposed for Debugging
- **Location:** All detection methods
- **Problem:** Score values computed but not logged when card detected
- **Recommendation:** Make debug logging configurable, add confidence scores to logs

### 19. **Type Hints Incomplete** (LOW)
**Files:** `bot_logic.py`, `action_executor.py`

#### Issue 19A: Union Types Use Pipe Instead of Union[]
- **Location:** [action_executor.py](action_executor.py#L31), [table_parser.py](table_parser.py#L93)
- **Code:**
  ```python
  self._ocr_cache: Dict[tuple, list[str]] = {}  # Should be Dict[Tuple[...], List[str]]
  self._last_action_state: Dict[str, object] | None = None  # Pre-3.10 incompatible
  ```
- **Impact:** Type checking warnings in older Python versions
- **Recommendation:** Use `from typing import Union` and `Union[Dict[...], None]` or upgrade to Python 3.10+

### 20. **Configuration Not Version-Controlled** (LOW)
**Files:** `config.py`

#### Issue 20A: No Configuration Schema/Validation
- **Location:** [config.py](config.py#L1-250)
- **Problem:** No type checking or validation of config values
- **Example:** If someone sets `LIVE_CONFIG['analysis_interval'] = "0.5"` (string), detection will fail
- **Recommendation:** Use Pydantic dataclasses for config validation

#### Issue 20B: Magic Numbers in config.py Not Documented
- **Location:** [config.py](config.py#L148-195)
- **Example:** Reference sizes (1935, 1369) have no explanation
- **Recommendation:** Add comments explaining how these were derived

---

## SUMMARY OF FINDINGS

### Statistics
- **Critical Issues:** 5 (thresholds, OCR setup, templates, layout detection, boundaries)
- **High Severity:** 7 (ROI calculations, surface extraction, cascade failures, HSV thresholds, caching)
- **Medium Severity:** 5 (text parsing, game state validation, error recovery, template verification, OCR integration)
- **Low Severity:** 3 (logging, type hints, configuration validation)

### Root Cause Analysis

| Issue Category | Count | Root Cause |
|---|---|---|
| Threshold Values | 8 | Thresholds copied from old system without re-tuning |
| Layout-Specific Logic | 4 | No abstraction for different table layouts |
| Template Management | 4 | No verification that templates are complete |
| Error Handling | 4 | Silent failures, no diagnostics |
| Configuration | 3 | Hardcoded values instead of config-driven |

### Recommended Fix Priority

**Phase 1 (Must Fix - Week 1):**
1. Reduce card match threshold from 0.82 to 0.70 [Issue #1B]
2. Add Tesseract config to CardDetector [Issue #3A]
3. Validate all 52 card templates load successfully [Issue #4B]
4. Add bounds checking to bot_logic.py card coordinates [Issue #8B]

**Phase 2 (Should Fix - Week 2):**
5. Increase rank/suit corner thresholds [Issues #2A, #2B]
6. Implement layout-specific detection methods [Issue #10A]
7. Add duplicate card validation [Issue #14A]
8. Implement error retry mechanism [Issue #15A]

**Phase 3 (Nice to Have - Week 3):**
9. Calibrate HSV color space thresholds [Issue #11A]
10. Improve OCR configuration [Issue #17A]
11. Add detailed confidence-based logging [Issue #18A]
12. Validate configuration with Pydantic [Issue #20B]

---

## TEST RECOMMENDATIONS

1. **Create test cases for edge conditions:**
   - Cards at image boundaries
   - Rotated/skewed cards
   - Low-light conditions
   - Different card back colors

2. **Add regression tests for each threshold:**
   - Create test set with 100+ real screenshots
   - Document expected detections for each
   - Run before/after threshold changes

3. **Implement continuous detection monitoring:**
   - Log accuracy metrics per card position
   - Alert if detection success rate drops below 95%
   - A/B test threshold changes

---

## FILE LOCATIONS REFERENCE

| File | Issues | Lines |
|---|---|---|
| detectors/card_detector.py | 1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18 | Critical: 49-54, 103, 1080-1100, 2044-2053, 2088-2095, 920-970, 1525-1530, 1903, 887-903 |
| config.py | 6, 20 | 148-195, 202-228 |
| bot_logic.py | 8, 13, 14, 15 | 48, 94-155, 172-190 |
| detectors/table_parser.py | 13 | 25, 283-284 |
| action_executor.py | 19 | 31, 93 |

---

**End of Report**
