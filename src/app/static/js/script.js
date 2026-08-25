// Tab Navigation
function switchTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
    document.getElementById(tabId).style.display = 'block';
    
    document.querySelectorAll('.list-group-item').forEach(el => el.classList.remove('active-tab'));
    event.target.classList.add('active-tab');
}

// Preset Payloads
const presets = {
    overachiever: { Hours_Studied: 35, Attendance: 98, Previous_Scores: 95, Sleep_Hours: 8, Tutoring_Sessions: 2, Physical_Activity: 5 },
    average: { Hours_Studied: 15, Attendance: 80, Previous_Scores: 75, Sleep_Hours: 6, Tutoring_Sessions: 0, Physical_Activity: 2 },
    atRisk: { Hours_Studied: 4, Attendance: 60, Previous_Scores: 55, Sleep_Hours: 4, Tutoring_Sessions: 0, Physical_Activity: 0 }
};

function applyPreset(type) {
    const data = presets[type];
    for (const [key, value] of Object.entries(data)) {
        document.getElementById(key).value = value;
    }
}

// Chart.js Setup for Models Tab
// Chart.js Setup for Models Tab (Dynamic Fetch)
document.addEventListener("DOMContentLoaded", async () => {
    try {
        const response = await fetch('/api/model-stats');
        const data = await response.json();
        
        const ctx = document.getElementById('modelChart').getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'RMSE (Lower is Better)',
                    data: data.rmse,
                    backgroundColor: 'rgba(56, 189, 248, 0.6)',
                    borderColor: '#38bdf8',
                    borderWidth: 1
                }]
            },
            options: { 
                responsive: true, 
                scales: { 
                    y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.1)' } }, 
                    x: { grid: { display: false } } 
                }, 
                color: '#fff',
                plugins: {
                    legend: { labels: { color: '#fff' } }
                }
            }
        });
    } catch (error) {
        console.error("Error loading chart data:", error);
    }
});

// AJAX Form Submission & SHAP Translator
document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = {
        Hours_Studied: document.getElementById('Hours_Studied').value,
        Attendance: document.getElementById('Attendance').value,
        Previous_Scores: document.getElementById('Previous_Scores').value,
        Sleep_Hours: document.getElementById('Sleep_Hours').value,
        Tutoring_Sessions: document.getElementById('Tutoring_Sessions').value,
        Physical_Activity: document.getElementById('Physical_Activity').value,
        // Defaulting categoricals for the frontend demo
        Parental_Involvement: "Medium", Access_to_Resources: "Medium", Motivation_Level: "Medium",
        Family_Income: "Medium", Teacher_Quality: "Medium", Peer_Influence: "Neutral",
        School_Type: "Public", Internet_Access: "Yes", Extracurricular_Activities: "Yes",
        Learning_Disabilities: "No", Parental_Education_Level: "High School", Distance_from_Home: "Near", Gender: "Male"
    };

    const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
    });
    
    const result = await response.json();
    
    // Populate Results
    document.getElementById('resScore').textContent = result.score;
    
    const riskBadge = document.getElementById('resRisk');
    riskBadge.textContent = result.risk;
    riskBadge.className = `badge fs-6 ${result.risk === 'High Risk' ? 'bg-danger' : result.risk === 'Moderate Risk' ? 'bg-warning text-dark' : 'bg-success'}`;
    
    // Human-Readable SHAP Translator
    let shapHtml = '';
    result.features.forEach(f => {
        const impact = f.contribution > 0 ? 'boosted' : 'dragged down';
        const cssClass = f.contribution > 0 ? 'shap-positive' : 'shap-negative';
        shapHtml += `<p class="mb-1 border-bottom border-secondary pb-1">
            Because <strong>${f.name}</strong> was recorded as ${f.value}, it 
            <span class="${cssClass}">${impact} the projection by ${Math.abs(f.contribution)} points.</span>
        </p>`;
    });
    document.getElementById('resShap').innerHTML = shapHtml;
    
    // Directives
    let recsHtml = '';
    result.recommendations.forEach(r => recsHtml += `<li class="mb-2 text-light">${r}</li>`);
    document.getElementById('resRecs').innerHTML = recsHtml;
    
    // Fade in results
    document.getElementById('resultsPanel').style.opacity = 1;
});