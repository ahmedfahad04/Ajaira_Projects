document.addEventListener('DOMContentLoaded', () => {
    const searchBtn = document.getElementById('search-btn');

    searchBtn.addEventListener('click', async () => {
        const confName = document.getElementById('conf-name-input').value.trim();
        console.log(`User entered: ${confName}`);

        if (!confName) {
            alert('Please enter a conference name');
            return;
        }

        console.log(`Searching for: ${confName}`);
        const rank = await getConferenceRank(confName);
        console.log(`Rank found: ${rank}`);

        document.getElementById('conf-name').textContent = `Conference name: ${confName}`;
        document.getElementById('conf-rank').innerHTML = rank || 'Rank info not found';
    });
});

// Function to get the conference rank from the CORE portal
async function getConferenceRank(confName) {
    const baseUrl = "https://portal.core.edu.au/conf-ranks/";
    const searchUrl = `${baseUrl}?search=${encodeURIComponent(confName)}&by=all&source=CORE2023&sort=atitle&page=1`;

    console.log(`Fetching URL: ${searchUrl}`);

    try {
        const res = await fetch(searchUrl);
        console.log(`Response status: ${res.status}`);

        if (!res.ok) {
            throw new Error('Failed to fetch data');
        }

        const htmlText = await res.text();
        console.log('HTML response received');

        const parser = new DOMParser();
        const doc = parser.parseFromString(htmlText, "text/html");

        const tableRows = doc.querySelectorAll("table tr");
        let results = [];
        console.log(`Found ${tableRows.length} table rows`);

        tableRows.forEach((row, index) => {
            if (index === 0) return; // skip header

            const cells = row.querySelectorAll("td");
            if (cells.length >= 9) {
                results.push({
                    title: cells[0].textContent.trim(),
                    acronym: cells[1].textContent.trim(),
                    source: cells[2].textContent.trim(),
                    rank: cells[3].textContent.trim(),
                    note: cells[4].textContent.trim(),
                    dblp: cells[5]?.querySelector("a")?.href || "N/A",
                    for: cells[6].textContent.trim(),
                    comments: cells[7].textContent.trim(),
                    rating: cells[8].textContent.trim()
                });
            }
        });

        console.log(`Parsed results: ${results.length}`);

        if (results.length === 0) {
            return "No results found.";
        }

        // Filter for exact acronym match
        const exactMatch = results.find(r => r.acronym.toLowerCase() === confName.toLowerCase());
        const chosen = exactMatch || results[0];

        console.log(`Chosen result: ${JSON.stringify(chosen)}`);

        // Stylish Output
        return `
        <div style="font-family: Arial, sans-serif; font-size: 14px; line-height: 1.6;">
          <p><strong style="font-size: 16px;">Title:</strong> ${chosen.title}</p>
          <p><strong style="font-size: 16px;">Acronym:</strong> ${chosen.acronym}</p>
          <p><strong style="font-size: 16px; color: #1d72b8;">Rank:</strong> <span style="font-weight: bold; color: #ff6347; font-size: 18px">${chosen.rank}</span></p>
          <p><strong style="font-size: 16px;">Note:</strong> ${chosen.note}</p>
          <p><strong style="font-size: 16px;">DBLP:</strong> <a href="${chosen.dblp}" target="_blank" style="color: #ff6347;">${chosen.dblp}</a></p>
          <p><strong style="font-size: 16px;">Average Rating:</strong> ${chosen.rating}</p>
        </div>
      `;

    } catch (err) {
        console.error("Failed to fetch rank:", err);
        return "Error fetching data.";
    }
}
