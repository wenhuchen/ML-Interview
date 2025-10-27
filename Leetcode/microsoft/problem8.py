from collections import Counter

def maxPalindromeLength(s1, s2):
    count1 = Counter(s1)
    count2 = Counter(s2)
    total_count = count1 + count2
    length = 0
    odd_found = False
    
    for count in total_count.values():
        if count % 2 == 0:
            length += count
        else:
            length += count - 1
            odd_found = True
    
    if odd_found:
        length += 1
    
    return length

# Example usage
if __name__ == "__main__":
    print(maxPalindromeLength("ababc", "def"))  # Output: 5
    print(maxPalindromeLength("abc", "abbc"))  # Output: 4