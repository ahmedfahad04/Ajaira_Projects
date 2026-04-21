function find_substring_matches(str, substr) {
    let count = 0;
    for (let i = 0; i <= str.length - substr.length; i++) {
        if (str.substr(i, substr.length) === substr) {
            count++;
        }
    }
    return count;
}
