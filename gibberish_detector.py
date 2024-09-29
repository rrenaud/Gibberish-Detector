# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 17:43:33 2024

@author: viraljansari_zeptono
"""

import re

def is_gibberish(address):
    # Rule 1: Check for common words found in addresses
    common_words = ['street', 'road', 'floor', 'block', 'metro', 'station', 'nagar', 'colony', 'city', 'town', 'village', 'district', 'apartment', 'building', 'phase', 'mandir', 'library']
    if any(word in address.lower() for word in common_words):
        return False

    # Rule 2: Check for too many consecutive consonants (a common sign of gibberish)
    if re.search(r'[^aeiou]{5,}', address.lower()):
        return True

    # Rule 3: Check if the address consists mostly of words that look like gibberish (random letter sequences)
    words = address.split()
    gibberish_count = sum(1 for word in words if re.search(r'[bcdfghjklmnpqrstvwxyz]{4,}', word.lower()) and len(word) > 4)
    if gibberish_count >= len(words) // 2:
        return True

    return False
