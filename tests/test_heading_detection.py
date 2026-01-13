"""
Test script to verify heading detection improvements.
Tests various heading formats found in the STALKER rulebook.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'parsers'))

from pdf_parser import is_all_caps_section, is_title_case_heading

def test_heading_detection():
    """Test various heading formats."""
    
    print("=" * 70)
    print("TESTING HEADING DETECTION")
    print("=" * 70)
    
    # Test cases for ALL CAPS sections
    all_caps_tests = [
        ("ZONE NAVIGATION", True, "Simple all caps"),
        ("TOKENS", True, "Single word heading"),
        ("PLAYING THE MISSION", True, "Multi-word heading"),
        ("(BASIC) ' (RANGE 0)", False, "Malformed with game mechanics"),
        ("LIGHT WOUND LIGHT WOUND", False, "Card description, not heading"),
        ("TOHIT TORSO", False, "Card ability, not heading"),
        ("POST MISSION CLEANUP", True, "Valid multi-word heading"),
        ("GAME ROUND STRUCTURE", True, "Valid heading"),
        ("3 Shotgun, 3 Rifle, 2 MG", False, "Component list"),
        ("STALKER ATTACK SEQUENCE", True, "Valid heading"),
        ("INSTANT TRIGGERS AND ADDITIONAL RULES", True, "Long heading"),
        ("(6 STARING (4 )", False, "Malformed parenthetical"),
    ]
    
    # Test cases for Title Case subsections
    title_case_tests = [
        ("Mission Setup", True, "Simple title case"),
        ("Stalker Movement Rules", True, "Multi-word title case"),
        ("Line of Sight", True, "Title case with 'of'"),
        ("| MissionMap |", True, "Pipe-delimited CamelCase heading"),
        ("| Playingthe Mission |", True, "Pipe-delimited mixed case heading"),
        ("MissionMap", True, "CamelCase heading"),
        ("Playingthe Mission", True, "Mixed case heading"),
        ("Marking Locations", True, "Subsection heading"),
        ("Stickers", True, "Single word subsection"),
        ("Double Faced Cards", True, "Multi-word subsection"),
        ("Buying and Selling", True, "Subsection with 'and'"),
        ("Flipping an Environment Card", True, "Longer subsection heading"),
        ("Editor: Matt Click, Daniel Morley", False, "Credits with colon"),
        ("Game Design: Pawet Samborski", False, "Credits with colon"),
    ]
    
    # Run tests
    print("\n" + "=" * 70)
    print("ALL CAPS SECTION TESTS")
    print("=" * 70)
    passed = 0
    failed = 0
    for text, expected, description in all_caps_tests:
        result = is_all_caps_section(text)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"{status} | {description:40} | '{text[:50]}'")
        if result != expected:
            print(f"       Expected: {expected}, Got: {result}")
    
    print(f"\nAll Caps Tests: {passed} passed, {failed} failed")
    
    print("\n" + "=" * 70)
    print("TITLE CASE HEADING TESTS (Sections & Subsections)")
    print("=" * 70)
    passed = 0
    failed = 0
    for text, expected, description in title_case_tests:
        result = is_title_case_heading(text)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"{status} | {description:40} | '{text[:50]}'")
        if result != expected:
            print(f"       Expected: {expected}, Got: {result}")
    
    print(f"\nTitle Case Tests: {passed} passed, {failed} failed")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    test_heading_detection()
