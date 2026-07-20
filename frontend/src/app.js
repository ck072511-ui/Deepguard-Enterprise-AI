/* ============================================================================
   DeepGuard Single Page Application (SPA) Controller Router
   ============================================================================ */

import { DeepGuardAPIClient } from './api.js';
import { renderDashboard } from './pages/dashboard.js';
import { renderImageDetection, renderVideoDetection } from './pages/detection.js';
import { renderWebcamDetection } from './pages/webcam.js';
import { renderHistory } from './pages/history.js';
import { renderAnalytics } from './pages/analytics.js';
import { renderMetrics } from './pages/metrics.js';
import { renderProfile } from './pages/profile.js';
import { renderSettings } from './pages/settings.js';

// Initialize API Client connection
const apiClient = new DeepGuardAPIClient();

// Page Container Elements
const pageContentContainer = document.getElementById("page-content");
const pageTitleElement = document.getElementById("current-page-title");
const topbarModelName = document.getElementById("topbar-model-name");

// Navigation mapping dictionary
const pages = {
    'dashboard': { title: 'Dashboard System Telemetry', render: renderDashboard },
    'image-detect': { title: 'Image Deepfake Analysis', render: renderImageDetection },
    'video-detect': { title: 'Video Frame Sequence Analysis', render: renderVideoDetection },
    'webcam': { title: 'Real Time Video Stream Scanner', render: renderWebcamDetection },
    'history': { title: 'Prediction History Log Registry', render: renderHistory },
    'analytics': { title: 'Platform Inference Analytics', render: renderAnalytics },
    'metrics': { title: 'Model Classification Performance', render: renderMetrics },
    'profile': { title: 'User Account Profile', render: renderProfile },
    'settings': { title: 'Platform Control Settings', render: renderSettings }
};

// Global Toast Alert Helper
window.showToast = function(message, type = 'info') {
    const container = document.getElementById("toast-container");
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    
    // Choose icon
    const iconName = type === 'success' ? 'check-circle' : type === 'error' ? 'alert-octagon' : 'info';
    toast.innerHTML = `
        <i data-lucide="${iconName}"></i>
        <span>${message}</span>
    `;
    
    container.appendChild(toast);
    lucide.createIcons();
    
    // Animate in
    setTimeout(() => toast.classList.add("show"), 10);
    
    // Terminate and remove
    setTimeout(() => {
        toast.classList.remove("show");
        setTimeout(() => toast.remove(), 300);
    }, 4000);
};

// Routing navigation controller
function navigateToPage(pageKey) {
    const page = pages[pageKey];
    if (!page) return;

    // Update active nav items
    document.querySelectorAll(".nav-item").forEach(item => {
        if (item.getAttribute("data-page") === pageKey) {
            item.classList.add("active");
        } else {
            item.classList.remove("active");
        }
    });

    // Update header labels
    pageTitleElement.textContent = page.title;

    // Clear and render new page contents
    pageContentContainer.innerHTML = `<div class="skeleton" style="width: 100%; height: 380px; border-radius: var(--radius-lg);"></div>`;
    
    setTimeout(() => {
        page.render(apiClient, pageContentContainer)
            .then(() => {
                lucide.createIcons();
            })
            .catch(error => {
                console.error(`Error rendering page ${pageKey}:`, error);
                pageContentContainer.innerHTML = `
                    <div class="glass-card animate-fade-in" style="margin-top: 20px; display: flex; flex-direction: column; align-items: center; gap: 16px; text-align: center; padding: 40px 24px; max-width: 600px; margin-left: auto; margin-right: auto; border-color: rgba(239, 68, 68, 0.25); box-shadow: 0 0 24px rgba(239, 68, 68, 0.05);">
                        <i data-lucide="alert-octagon" style="width: 48px; height: 48px; color: var(--color-error);"></i>
                        <div>
                            <h3 style="font-weight: 700; margin-bottom: 8px; color: var(--text-primary);">Section Loading Failed</h3>
                            <p style="color: var(--text-muted); font-size: 0.9rem; max-width: 400px; margin: 0 auto; line-height: 1.5;">
                                DeepGuard was unable to load this section. This usually happens when the API server is offline, unreachable, or requires authentication.
                            </p>
                            <p style="color: var(--color-error); font-family: var(--font-mono); font-size: 0.8rem; margin-top: 16px; background: rgba(239, 68, 68, 0.05); padding: 8px 12px; border-radius: var(--radius-sm); display: inline-block; border: 1px solid rgba(239, 68, 68, 0.15);">
                                Details: ${error.message || error}
                            </p>
                        </div>
                        <button class="btn-primary" onclick="window.location.reload()" style="margin-top: 8px; font-size: 0.875rem; padding: 10px 18px;">
                            <i data-lucide="refresh-cw"></i> Retry Connection
                        </button>
                    </div>
                `;
                lucide.createIcons();
            });
    }, 150);
}

// Sidebar click triggers
document.querySelectorAll(".nav-item").forEach(item => {
    item.addEventListener("click", () => {
        const pageKey = item.getAttribute("data-page");
        navigateToPage(pageKey);
    });
});

// Topbar user icon redirect trigger
const profileTrigger = document.getElementById("topbar-profile-btn");
if (profileTrigger) {
    profileTrigger.addEventListener("click", () => navigateToPage("profile"));
}

// Theme Selector controls
document.querySelectorAll(".theme-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        const selectedTheme = btn.getAttribute("data-theme");
        
        // Remove active class from all theme selectors
        document.querySelectorAll(".theme-btn").forEach(b => b.classList.remove("active"));
        
        // Set new active selection
        btn.classList.add("active");
        
        // Update root layout theme variables attributes
        document.documentElement.setAttribute("data-theme", selectedTheme);
        
        window.showToast(`Theme switched to ${btn.title}.`, "info");
    });
});

// Setup health check routine
function updateApiStatus(isOnline, message = null) {
    const statusBadge = document.getElementById("api-status-badge");
    const statusText = statusBadge.querySelector(".status-text");

    statusBadge.className = `api-status-badge ${isOnline ? "online" : "offline"}`;
    statusText.textContent = `API: ${isOnline ? "Online" : (message || "Offline")}`;

    if (!isOnline && topbarModelName) {
        topbarModelName.textContent = "Unavailable";
    }
}

async function runHealthCheck() {
    try {
        const health = await apiClient.checkHealth();
        const isOnline = health?.status === "healthy";

        if (isOnline) {
            updateApiStatus(true);
            const models = await apiClient.listModels();
            const activeModel = models.find(m => m.active);
            if (activeModel && topbarModelName) {
                topbarModelName.textContent = activeModel.name;
            }
        } else {
            updateApiStatus(false, "Degraded");
        }
    } catch (error) {
        console.error("Health check failed:", error);
        updateApiStatus(false);
    }
}

window.addEventListener("online", () => {
    if (window.showToast) window.showToast("Network connection restored. Rechecking API...", "success");
    runHealthCheck();
});

window.addEventListener("offline", () => {
    if (window.showToast) window.showToast("You are offline. Some features may be unavailable.", "error");
    updateApiStatus(false, "Offline");
});

// Initialization on load
document.addEventListener("DOMContentLoaded", () => {
    // Perform initial API checking
    runHealthCheck();
    
    // Start interval health probes every 15s
    setInterval(runHealthCheck, 15000);

    // Route to default page
    navigateToPage("dashboard");
});
