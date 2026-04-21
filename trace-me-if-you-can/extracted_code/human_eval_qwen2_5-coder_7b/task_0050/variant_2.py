function shift_chars_left_by_five(s) {
    return s.split('').map(c => String.fromCharCode((c.charCodeAt(0) - 5 - 'a'.charCodeAt(0)) % 26 + 'a'.charCodeAt(0))).join('');
}
