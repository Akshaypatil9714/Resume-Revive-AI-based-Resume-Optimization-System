async function automateProcess() {
    let currentPage = 1;
    const totalPages = 10;
    let allJobs = [];
        const jobCards = document.querySelectorAll('div.job_seen_beacon');
        for (let i = 0; i < jobCards.length; i++) {
            let div = jobCards.item(i);
            const jobTitleElement = div.querySelector('h2.jobTitle a');
            if (jobTitleElement) {
                jobTitleElement.focus();  // Focus the element
                jobTitleElement.click();  // Click the job title to load details
                await new Promise(r => setTimeout(r, 2000));  // Wait for details to load
            }

            const jobTitle = jobTitleElement ? jobTitleElement.textContent.trim() : 'No title found';
            const jobDescription = document.getElementById('jobDescriptionText') ? document.getElementById('jobDescriptionText').innerText.trim() : 'No description found';

            allJobs.push({ jobTitle, jobDescription });
        }

        downloadCSV(allJobs);

        await new Promise(r => setTimeout(r, 3000)); // Simulate delay for page load


    console.log('Automation completed.');
}

function convertToCSV(objArray) {
    const array = typeof objArray !== 'object' ? JSON.parse(objArray) : objArray;
    let str = 'Job Title,Job Description\n';

    for (let i = 0; i < array.length; i++) {
        let line = '';
        for (let index in array[i]) {
            if (line !== '') line += ',';

            line += `"${array[i][index].replace(/"/g, '""')}"`; // Handle quotes
        }

        str += line + '\r\n';
    }

    return str;
}

function downloadCSV(jobs) {
    let csvData = convertToCSV(jobs);
    let blob = new Blob([csvData], { type: 'text/csv;charset=utf-8;' });
    let url = URL.createObjectURL(blob);
    let link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", "jobs.csv");
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

automateProcess();