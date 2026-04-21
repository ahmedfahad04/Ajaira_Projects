function getSubstringFrequency(text, sub) {
    let frequency = 0;
    for (let k = 0; k <= text.length - sub.length; k++) {
        if (text.substring(k, k + sub.length) === sub) {
            frequency++;
        }
    }
    return frequency;
}
