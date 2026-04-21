const substringCount = (str, sub) => {
    let occurrences = 0;
    for (let j = 0; j <= str.length - sub.length; j++) {
        if (str.slice(j, j + sub.length) === sub) {
            occurrences++;
        }
    }
    return occurrences;
};
