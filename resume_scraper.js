async function automateProcess() {
    let currentPage = 1;
    const totalPages = 10;
    const keywords = ["software engineer", "software developer", "data science"]; // Add more keywords as needed

    while (currentPage <= totalPages) {
        let xpath = "//div[@data-cauto-id='candidate-row']";
        let divs = document.evaluate(xpath, document, null, XPathResult.ORDERED_NODE_SNAPSHOT_TYPE, null);
        for (let i = 0; i < divs.snapshotLength; i++) {
            let div = divs.snapshotItem(i);
            // Check if the resume contains any of the keywords
            const containsKeyword = keywords.some(keyword => div.textContent.toLowerCase().includes(keyword.toLowerCase()));
            if (containsKeyword) {
                div.click();
                await new Promise(r => setTimeout(r, 5000));

                let downloadButtonXpath = "//*[@data-cauto-id='download_resume_action']";
                let downloadButtonResult = document.evaluate(downloadButtonXpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
                if (downloadButtonResult.singleNodeValue) {
                    downloadButtonResult.singleNodeValue.click();
                    await new Promise(r => setTimeout(r, 5000));
                }
            }
        }

        // Navigate to the next page
        let nextButtonXpath = "//*[text()='Next']";
        let nextButtonResult = document.evaluate(nextButtonXpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
        if (!nextButtonResult.singleNodeValue) break; 
        nextButtonResult.singleNodeValue.click();
        await new Promise(r => setTimeout(r, 3000));
        currentPage++;
    }

    console.log('Automation completed.');
}

automateProcess();
