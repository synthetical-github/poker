#!/usr/bin/env python3
"""
Comprehensive Card Detection Diagnostic Tool
Tests all aspects of the poker card detection system to identify issues.
"""

import cv2
import numpy as np
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("CardDetectionDiagnostic")

class CardDetectionDiagnostic:
    def __init__(self):
        self.results = {}
        self.issues = []
        
    def test_template_loading(self) -> bool:
        """Test if card templates are properly loaded."""
        logger.info("=" * 60)
        logger.info("TEST 1: Template Loading")
        logger.info("=" * 60)
        
        try:
            from detectors.card_detector import CardDetector
            detector = CardDetector()
            
            # Check card templates
            card_count = len(detector.card_templates)
            logger.info(f"✓ Card templates loaded: {card_count}")
            
            if card_count == 0:
                logger.error("✗ CRITICAL: No card templates loaded!")
                self.issues.append("No card templates found - check CARD_TEMPLATES_DIR in config.py")
                return False
            
            # Check rank and suit counts
            hero_ranks = len(detector.hero_rank_templates)
            hero_suits = len(detector.hero_suit_templates)
            board_ranks = len(detector.board_rank_templates)
            board_suits = len(detector.board_suit_templates)
            
            logger.info(f"  Hero rank templates: {hero_ranks}")
            logger.info(f"  Hero suit templates: {hero_suits}")
            logger.info(f"  Board rank templates: {board_ranks}")
            logger.info(f"  Board suit templates: {board_suits}")
            
            expected_ranks = 13  # A, 2-9, T, J, Q, K
            expected_suits = 4   # C, D, H, S
            
            if hero_ranks < expected_ranks * 0.5:
                logger.warning(f"  ⚠ Hero rank templates incomplete: {hero_ranks}/{expected_ranks}")
                self.issues.append(f"Hero rank templates incomplete: {hero_ranks}/{expected_ranks}")
            
            if hero_suits < expected_suits * 0.5:
                logger.warning(f"  ⚠ Hero suit templates incomplete: {hero_suits}/{expected_suits}")
                self.issues.append(f"Hero suit templates incomplete: {hero_suits}/{expected_suits}")
            
            logger.info("✓ Template loading test PASSED\n")
            return True
            
        except Exception as e:
            logger.error(f"✗ Template loading test FAILED: {e}")
            self.issues.append(f"Template loading failed: {str(e)}")
            return False
    
    def test_threshold_values(self) -> bool:
        """Test if threshold values are reasonable."""
        logger.info("=" * 60)
        logger.info("TEST 2: Threshold Configuration")
        logger.info("=" * 60)
        
        try:
            from detectors.card_detector import CardDetector
            detector = CardDetector()
            
            logger.info(f"Card match threshold: {detector.min_card_match_threshold}")
            logger.info(f"Card match gap: {detector.min_card_match_gap}")
            logger.info(f"Rank match threshold: {detector.min_rank_match_threshold}")
            logger.info(f"Corner rank threshold: {detector.min_corner_rank_match_threshold}")
            logger.info(f"Corner suit threshold: {detector.min_corner_suit_match_threshold}")
            
            # Check for overly strict thresholds
            if detector.min_card_match_threshold > 0.80:
                logger.warning(f"  ⚠ Card match threshold ({detector.min_card_match_threshold}) may be too strict")
                self.issues.append(f"Card match threshold {detector.min_card_match_threshold} is too high (>0.80)")
            
            if detector.min_rank_match_threshold > 0.60:
                logger.warning(f"  ⚠ Rank match threshold ({detector.min_rank_match_threshold}) may be too strict")
                self.issues.append(f"Rank threshold {detector.min_rank_match_threshold} is too high (>0.60)")
            
            logger.info("✓ Threshold test PASSED\n")
            return True
            
        except Exception as e:
            logger.error(f"✗ Threshold test FAILED: {e}\n")
            self.issues.append(f"Threshold test failed: {str(e)}")
            return False
    
    def test_tesseract_availability(self) -> bool:
        """Test if Tesseract OCR is available."""
        logger.info("=" * 60)
        logger.info("TEST 3: Tesseract OCR Availability")
        logger.info("=" * 60)
        
        try:
            import pytesseract
            # Try to get version
            try:
                result = pytesseract.get_tesseract_version()
                logger.info(f"✓ Tesseract found: {result}")
                logger.info("✓ Tesseract test PASSED\n")
                return True
            except Exception as e:
                logger.warning(f"⚠ Tesseract not fully accessible: {e}")
                logger.info("  Card detection will rely on template matching only (fallback mode)")
                logger.info("✓ Tesseract test PASSED (with fallback)\n")
                return True
                
        except Exception as e:
            logger.warning(f"⚠ Tesseract test WARNING: {e}")
            logger.info("  This is OK if using template matching only\n")
            return True
    
    def test_config_paths(self) -> bool:
        """Test if configuration paths are valid."""
        logger.info("=" * 60)
        logger.info("TEST 4: Configuration Paths")
        logger.info("=" * 60)
        
        try:
            from config import ASSETS_DIR, CARD_TEMPLATES_DIR
            
            logger.info(f"ASSETS_DIR: {ASSETS_DIR}")
            logger.info(f"CARD_TEMPLATES_DIR: {CARD_TEMPLATES_DIR}")
            
            if not os.path.isdir(ASSETS_DIR):
                logger.error(f"✗ ASSETS_DIR does not exist: {ASSETS_DIR}")
                self.issues.append(f"ASSETS_DIR not found: {ASSETS_DIR}")
                return False
            
            if not os.path.isdir(CARD_TEMPLATES_DIR):
                logger.error(f"✗ CARD_TEMPLATES_DIR does not exist: {CARD_TEMPLATES_DIR}")
                self.issues.append(f"CARD_TEMPLATES_DIR not found: {CARD_TEMPLATES_DIR}")
                return False
            
            # Count template files
            template_files = [f for f in os.listdir(CARD_TEMPLATES_DIR) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            logger.info(f"✓ Found {len(template_files)} template files")
            
            if len(template_files) < 20:
                logger.warning(f"  ⚠ Only {len(template_files)} templates found (expect ~52)")
                self.issues.append(f"Low template count: {len(template_files)}/52")
            
            logger.info("✓ Config paths test PASSED\n")
            return True
            
        except Exception as e:
            logger.error(f"✗ Config paths test FAILED: {e}\n")
            self.issues.append(f"Config paths test failed: {str(e)}")
            return False
    
    def test_image_processing(self) -> bool:
        """Test basic image processing functions."""
        logger.info("=" * 60)
        logger.info("TEST 5: Image Processing")
        logger.info("=" * 60)
        
        try:
            from detectors.card_detector import CardDetector
            detector = CardDetector()
            
            # Create a test image
            test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            logger.info(f"✓ Created test image: {test_image.shape}")
            
            # Test color space conversion
            gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            logger.info(f"✓ BGR to Grayscale conversion works")
            
            # Test HSV conversion
            hsv = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
            logger.info(f"✓ BGR to HSV conversion works")
            
            # Test symbol mask building
            mask = detector._build_symbol_mask(test_image)
            logger.info(f"✓ Symbol mask building works: {mask.shape}")
            
            logger.info("✓ Image processing test PASSED\n")
            return True
            
        except Exception as e:
            logger.error(f"✗ Image processing test FAILED: {e}\n")
            self.issues.append(f"Image processing failed: {str(e)}")
            return False
    
    def test_card_detection_flow(self) -> bool:
        """Test the complete card detection flow."""
        logger.info("=" * 60)
        logger.info("TEST 6: Card Detection Flow")
        logger.info("=" * 60)
        
        try:
            from detectors.card_detector import CardDetector
            from config import LIVE_CONFIG
            
            detector = CardDetector()
            
            # Create a test region
            test_region = (100, 100, 80, 120)  # x, y, w, h
            
            # Create test image
            test_image = np.random.randint(0, 256, (500, 700, 3), dtype=np.uint8)
            
            logger.info(f"Test region: {test_region}")
            logger.info(f"Test image shape: {test_image.shape}")
            
            # Test detect_cards_in_regions with a list
            regions = [test_region]
            results = detector.detect_cards_in_regions(test_image, regions, context="hero1")
            
            logger.info(f"✓ Card detection flow completed")
            logger.info(f"  Results: {results}")
            logger.info("✓ Card detection flow test PASSED\n")
            return True
            
        except Exception as e:
            logger.error(f"✗ Card detection flow test FAILED: {e}\n")
            self.issues.append(f"Card detection flow failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_all_tests(self) -> Tuple[int, int]:
        """Run all diagnostic tests."""
        logger.info("\n")
        logger.info("╔" + "=" * 58 + "╗")
        logger.info("║" + " POKER CARD DETECTION DIAGNOSTIC TOOL ".center(58) + "║")
        logger.info("╚" + "=" * 58 + "╝\n")
        
        tests = [
            ("Template Loading", self.test_template_loading),
            ("Threshold Configuration", self.test_threshold_values),
            ("Tesseract OCR", self.test_tesseract_availability),
            ("Configuration Paths", self.test_config_paths),
            ("Image Processing", self.test_image_processing),
            ("Card Detection Flow", self.test_card_detection_flow),
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Unexpected error in {test_name}: {e}\n")
                failed += 1
        
        # Summary
        logger.info("╔" + "=" * 58 + "╗")
        logger.info("║" + " DIAGNOSTIC SUMMARY ".center(58) + "║")
        logger.info("╠" + "=" * 58 + "╣")
        logger.info(f"║ Tests Passed: {passed:<48} ║")
        logger.info(f"║ Tests Failed: {failed:<48} ║")
        logger.info("╠" + "=" * 58 + "╣")
        
        if self.issues:
            logger.info("║ ISSUES FOUND:".ljust(59) + "║")
            for i, issue in enumerate(self.issues, 1):
                # Split long issues into multiple lines
                for line in self._wrap_text(issue, 54):
                    logger.info(f"║ {i}. {line:<54} ║")
                    i = " "
        else:
            logger.info("║ ✓ No issues found!".ljust(59) + "║")
        
        logger.info("╚" + "=" * 58 + "╝\n")
        
        return passed, failed
    
    def _wrap_text(self, text: str, width: int) -> List[str]:
        """Wrap text to fit within width."""
        lines = []
        while len(text) > width:
            # Find last space within width
            split_idx = text.rfind(' ', 0, width)
            if split_idx == -1:
                split_idx = width
            lines.append(text[:split_idx])
            text = text[split_idx:].lstrip()
        if text:
            lines.append(text)
        return lines

def main():
    """Run the diagnostic tool."""
    diagnostic = CardDetectionDiagnostic()
    passed, failed = diagnostic.run_all_tests()
    
    # Return appropriate exit code
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
