document.getElementById("predictForm").addEventListener("submit", async function (e) {
    e.preventDefault();

    const ticker = document.getElementById("ticker").value.toUpperCase();
    const startDate = document.getElementById("startDate").value;
    const endDate = document.getElementById("endDate").value;
    const futureDays = document.getElementById("futureDays").value;

    const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker, start_date: startDate, end_date: endDate, future_days: futureDays }),
    });

    const data = await response.json();
    displayResults(data);
    plotGraph(data.future_prices);
});

function displayResults(data) {
    let resultsDiv = document.getElementById("results");
    

    data.future_prices.forEach(item => {
        resultsDiv.innerHTML += `<li>${item.date}: ${item.price}</li>`;
    });
    resultsDiv.innerHTML = `<h2>Prediction Results</h2><p><strong>Accuracy:</strong> ${data.accuracy.toFixed(2)}%</p><h3>Future Prices:</h3><ul>`;
    resultsDiv.innerHTML += `</ul>`;
}

function plotGraph(futurePrices) {
    const ctx = document.getElementById("predictionChart").getContext("2d");

    // Extract dates and prices from API response
    const labels = futurePrices.map(item => item.date);
    const prices = futurePrices.map(item => item.price);

    // Remove previous chart instance if exists
    if (window.myChart) {
        window.myChart.destroy();
    }

    window.myChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [{
                label: "Predicted Stock Prices",
                data: prices,
                borderColor: "blue",
                fill: false,
                tension: 0.2
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: true },
                tooltip: { enabled: true }
            }
        }
    });
}
