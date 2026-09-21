let fraudOverviewChart = null;
let probabilityChart = null;

async function fetchJson(url, options = {}) {
    const res = await fetch(url, options);
    if (!res.ok) {
        const text = await res.text();
        throw new Error(text || res.statusText);
    }
    return res.json();
}

async function initDashboard() {
    try {
        const data = await fetchJson("/api/stats");
        const stats = data.stats;
        const recent = data.recent;

        document.getElementById("total-emails").textContent = stats.total_emails;
        document.getElementById("fraud-detected").textContent = stats.fraud_detected;
        document.getElementById("genuine-emails").textContent = stats.genuine_emails;

        const ctx = document.getElementById("fraudOverviewChart");
        if (ctx) {
            fraudOverviewChart = new Chart(ctx, {
                type: "bar",
                data: {
                    labels: ["Fraud", "Genuine"],
                    datasets: [
                        {
                            label: "Detections",
                            data: [stats.fraud_detected, stats.genuine_emails],
                            backgroundColor: ["#ff6b6b", "#4cd964"],
                        },
                    ],
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { display: false },
                    },
                    scales: {
                        x: {
                            ticks: { color: "#c2cffc" },
                            grid: { color: "rgba(255,255,255,0.05)" },
                        },
                        y: {
                            ticks: { color: "#c2cffc" },
                            grid: { color: "rgba(255,255,255,0.05)" },
                        },
                    },
                },
            });
        }

        const list = document.getElementById("recent-activity-list");
        if (list) {
            list.innerHTML = "";
            if (!recent.length) {
                list.innerHTML = "<li>No recent analyses yet.</li>";
            } else {
                recent.forEach((item) => {
                    const li = document.createElement("li");
                    const title = document.createElement("div");
                    title.textContent = item.subject || "N/A subject";
                    const meta = document.createElement("div");
                    meta.className = "activity-meta";

                    const badges = document.createElement("div");
                    const predBadge = document.createElement("span");
                    predBadge.className =
                        "badge " + (item.prediction.toLowerCase() === "fake" ? "fake" : "genuine");
                    predBadge.textContent = item.prediction;

                    const riskBadge = document.createElement("span");
                    riskBadge.className = "badge " + item.risk_level.toLowerCase();
                    riskBadge.textContent = item.risk_level;

                    badges.appendChild(predBadge);
                    badges.appendChild(riskBadge);

                    const ts = document.createElement("span");
                    ts.textContent = item.created_at;

                    meta.appendChild(badges);
                    meta.appendChild(ts);

                    li.appendChild(title);
                    li.appendChild(meta);
                    list.appendChild(li);
                });
            }
        }
    } catch (e) {
        // Soft fail on dashboard stats
        // console.error(e);
    }
}

function renderProbabilityChart(fraudProb, genuineProb) {
    const ctx = document.getElementById("probabilityChart");
    if (!ctx) return;
    if (probabilityChart) {
        probabilityChart.destroy();
    }
    probabilityChart = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: ["Fraud probability", "Genuine probability"],
            datasets: [
                {
                    data: [fraudProb, genuineProb],
                    backgroundColor: ["#ff6b6b", "#4cd964"],
                    borderWidth: 0,
                },
            ],
        },
        options: {
            cutout: "72%",
            plugins: {
                legend: {
                    display: false,
                },
            },
        },
    });

    const centerVal = Math.round(genuineProb);
    const donutValue = document.getElementById("donut-value");
    const donutCaption = document.getElementById("donut-caption");
    if (donutValue) donutValue.textContent = `${centerVal}%`;
    if (donutCaption) donutCaption.textContent = centerVal >= 50 ? "Genuine" : "Fraud";
}

async function initAnalysis() {
    const analyzeBtn = document.getElementById("analyze-btn");
    if (!analyzeBtn) return;

    analyzeBtn.addEventListener("click", async () => {
        const subjectEl = document.getElementById("subject");
        const textEl = document.getElementById("email_text");
        const fileEl = document.getElementById("file");
        const errorEl = document.getElementById("analysis-error");
        const warningEl = document.getElementById("analysis-warning");
        const resultSection = document.getElementById("result-section");
        const placeholder = document.getElementById("result-placeholder");

        errorEl.textContent = "";
        if (warningEl) {
            warningEl.textContent = "";
            warningEl.classList.add("hidden");
        }

        const emailText = textEl.value.trim();
        const file = fileEl.files[0];
        if (!emailText && !file) {
            errorEl.textContent = "Please paste email content or upload a file.";
            return;
        }

        const formData = new FormData();
        formData.append("subject", subjectEl.value || "");
        formData.append("email_text", emailText);
        if (file) {
            formData.append("file", file);
        }

        analyzeBtn.disabled = true;
        analyzeBtn.textContent = "Analyzing...";

        try {
            const res = await fetch("/api/analyze-email", {
                method: "POST",
                body: formData,
            });
            const data = await res.json();
            if (!res.ok) {
                // Show short-input messages as a softer warning if possible
                if (warningEl && data && data.error) {
                    warningEl.textContent = data.error;
                    warningEl.classList.remove("hidden");
                    return;
                }
                throw new Error((data && data.error) || "Failed to analyze email.");
            }

            placeholder.classList.add("hidden");
            resultSection.classList.remove("hidden");

            const predictionText = data.prediction;
            const riskLevel = data.risk_level;

            const pill = document.getElementById("prediction-pill");
            if (pill) {
                pill.textContent = predictionText;
                pill.className =
                    "prediction-pill " +
                    (predictionText && predictionText.toLowerCase() === "fake" ? "fake" : "genuine");
            }

            const riskEl = document.getElementById("result-risk");
            if (riskEl) {
                riskEl.textContent = riskLevel;
            }

            const riskFill = document.getElementById("risk-bar-fill");
            if (riskFill) {
                const fraudPct = typeof data.fraud_probability === "number" ? data.fraud_probability : 0;
                riskFill.style.width = `${fraudPct}%`;
                riskFill.className = "risk-bar-fill " + (riskLevel ? riskLevel.toLowerCase() : "");
            }

            const fraudPctEl = document.getElementById("fraud-pct");
            const genuinePctEl = document.getElementById("genuine-pct");
            if (fraudPctEl) fraudPctEl.textContent = `${data.fraud_probability}%`;
            if (genuinePctEl) genuinePctEl.textContent = `${data.genuine_probability}%`;

            const reasonsList = document.getElementById("result-reasons");
            reasonsList.innerHTML = "";
            (data.reasons || []).forEach((r) => {
                const li = document.createElement("li");
                li.textContent = r;
                reasonsList.appendChild(li);
            });

            const kwContainer = document.getElementById("result-keywords");
            kwContainer.innerHTML = "";
            (data.suspicious_keywords || []).forEach((kw) => {
                const span = document.createElement("span");
                span.className = "keyword-pill";
                span.textContent = kw;
                kwContainer.appendChild(span);
            });

            renderProbabilityChart(data.fraud_probability, data.genuine_probability);
        } catch (e) {
            errorEl.textContent = e.message;
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = "Analyze Email";
        }
    });
}

async function initReports() {
    const list = document.getElementById("reports-list");
    if (!list) return;

    async function loadReports() {
        const search = document.getElementById("search-input").value || "";
        const risk = document.getElementById("filter-risk").value || "";
        const pred = document.getElementById("filter-prediction").value || "";
        const params = new URLSearchParams();
        if (search) params.append("search", search);
        if (risk) params.append("risk_level", risk);
        if (pred) params.append("prediction", pred);

        const data = await fetchJson("/api/reports?" + params.toString());
        const reports = data.reports;
        list.innerHTML = "";
        if (!reports.length) {
            list.innerHTML = "<div class='empty-state'>No reports match the current filters.</div>";
            return;
        }

        reports.forEach((r) => {
            const card = document.createElement("div");
            card.className = "scan-card";
            const predClass = (r.prediction || "").toLowerCase() === "fake" ? "fake" : "genuine";
            const riskClass = (r.risk_level || "").toLowerCase();
            card.innerHTML = `
                <div class="scan-main">
                    <div class="scan-title">${r.subject || "N/A"}</div>
                    <div class="scan-meta">
                        <span class="muted">${r.email_id}</span>
                        <span class="muted">${r.created_at}</span>
                    </div>
                </div>
                <div class="scan-badges">
                    <span class="badge ${predClass}">${r.prediction}</span>
                    <span class="badge ${riskClass}">${r.risk_level}</span>
                </div>
            `;
            list.appendChild(card);
        });
    }

    document.getElementById("filter-apply").addEventListener("click", loadReports);
    document.getElementById("search-input").addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            loadReports();
        }
    });

    loadReports();
}

document.addEventListener("DOMContentLoaded", () => {
    const page = document.body.getAttribute("data-page");
    if (page === "dashboard") {
        initDashboard();
    } else if (page === "analysis") {
        initAnalysis();
    } else if (page === "reports") {
        initReports();
    }
});

