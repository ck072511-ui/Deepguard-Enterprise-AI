/* ============================================================================
   DeepGuard Frontend Page — User Profile
   ============================================================================ */

export async function renderProfile(apiClient, container) {
    const user = await apiClient.getCurrentUser();
    const avatarInitials = user.sub ? user.sub.split(' ').map(n => n[0]).join('').substring(0, 2).toUpperCase() : "AG";
    const authDisabledNotice = user.auth_disabled ? ` <span style="color: var(--color-warning); font-size: 0.75rem; display: block; margin-top: 4px;">⚠️ Authentication is disabled for local development</span>` : "";

    container.innerHTML = `
        <div class="animate-fade-in">
            <div class="glass-card" style="max-width: 720px; margin: 0 auto;">
                <div class="card-header" style="border-bottom: 1px solid var(--border-color); padding-bottom: 20px;">
                    <h3>Account Profile</h3>
                    <span class="card-header-sub">Developer credentials & security settings</span>
                </div>

                <div style="display: flex; gap: 32px; align-items: center; margin: 32px 0;">
                    <div style="background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-secondary) 100%); 
                                width: 120px; height: 120px; border-radius: 50%; 
                                display: flex; align-items: center; justify-content: center;
                                font-size: 3rem; color: #fff; box-shadow: 0 4px 20px var(--accent-glow); flex-shrink: 0;">
                        ${avatarInitials}
                    </div>
                    <div>
                        <h2 style="font-weight: 800; font-size: 1.5rem; color: var(--text-primary);">${user.sub}</h2>
                        <p style="color: var(--text-muted); font-size: 0.95rem; margin-top: 4px;">Role: ${user.role}</p>
                        <p style="color: var(--color-primary); font-size: 0.85rem; font-weight: 700; margin-top: 8px;">🛡️ Super Administrator</p>
                    </div>
                </div>

                <div class="form-group">
                    <label>E-mail Address</label>
                    <input type="text" class="form-control" value="${user.email}" disabled>
                </div>

                <div class="form-group">
                    <label>Active Access API Key</label>
                    <div style="display: flex; gap: 10px;">
                        <input type="password" class="form-control" value="${user.auth_disabled ? 'DEEPGUARD_AUTH_DISABLED=1 (Bypassed)' : user.api_key}" id="profile-api-key-input" style="font-family: var(--font-mono);" disabled>
                        <button class="btn-secondary" id="btn-toggle-key-reveal" style="padding: 12px 18px;">
                            <i data-lucide="eye" id="toggle-key-icon"></i>
                        </button>
                    </div>
                    <span style="font-size: 0.8rem; color: var(--text-muted); display: block; margin-top: 6px;">
                        Keep this key confidential. Never share or expose in production logs.${authDisabledNotice}
                    </span>
                </div>

                <div style="margin-top: 32px; border-top: 1px solid var(--border-color); padding-top: 24px; display: flex; justify-content: flex-end;">
                    <button class="btn-primary" id="btn-save-profile">
                        <i data-lucide="check"></i> Save Profiles Details
                    </button>
                </div>
            </div>
        </div>
    `;

    const keyInput = document.getElementById("profile-api-key-input");
    const revealBtn = document.getElementById("btn-toggle-key-reveal");
    const icon = document.getElementById("toggle-key-icon");
    const saveBtn = document.getElementById("btn-save-profile");
    let revealed = false;

    revealBtn.addEventListener("click", () => {
        revealed = !revealed;
        if (revealed) {
            keyInput.type = "text";
            revealBtn.innerHTML = `<i data-lucide="eye-off"></i>`;
        } else {
            keyInput.type = "password";
            revealBtn.innerHTML = `<i data-lucide="eye"></i>`;
        }
        lucide.createIcons();
    });

    saveBtn.addEventListener("click", () => {
        if (window.showToast) window.showToast("Profile details saved successfully.", "success");
    });

    lucide.createIcons();
}
