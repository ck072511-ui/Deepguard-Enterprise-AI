/* ============================================================================
   DeepGuard Frontend Page — Prediction History Registry
   ============================================================================ */

export async function renderHistory(apiClient, container) {
    let currentPage = 1;
    const pageSize = 10;
    let selectedMediaType = 'all';
    let selectedOutcome = 'all';
    let searchQuery = '';

    container.innerHTML = `
        <div class="animate-fade-in">
            <div class="glass-card" style="margin-bottom: 24px;">
                <div class="card-header">
                    <h3>Filter Log Registry</h3>
                    <span class="card-header-sub">Search through database records</span>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="history-search-input">Search Filename</label>
                        <input type="text" id="history-search-input" class="form-control" placeholder="🔍 Type filename to search...">
                    </div>
                    <div class="form-group">
                        <label for="history-type-select">Media Format</label>
                        <select id="history-type-select" class="form-control">
                            <option value="all">All Formats</option>
                            <option value="image">Images</option>
                            <option value="video">Videos</option>
                        </select>
                    </div>
                </div>

                <div class="form-row" style="margin-top: 10px;">
                    <div class="form-group">
                        <label for="history-outcome-select">Model Prediction</label>
                        <select id="history-outcome-select" class="form-control">
                            <option value="all">All Outcomes</option>
                            <option value="REAL">REAL</option>
                            <option value="FAKE">FAKE</option>
                        </select>
                    </div>
                    <div class="form-group" style="display: flex; align-items: flex-end; justify-content: flex-end;">
                        <button class="btn-primary" id="btn-apply-filters" style="width: 100%; justify-content: center;">
                            <i data-lucide="filter"></i> Apply Filters
                        </button>
                    </div>
                </div>
            </div>

            <!-- Log Table -->
            <div class="glass-card">
                <div class="scans-table-container">
                    <table class="table-scans">
                        <thead>
                            <tr>
                                <th>Record ID</th>
                                <th>Filename</th>
                                <th>Media Type</th>
                                <th>Outcome</th>
                                <th>Confidence</th>
                                <th>Faces</th>
                                <th>Timestamp</th>
                            </tr>
                        </thead>
                        <tbody id="history-table-tbody">
                            <!-- Table rows go here -->
                        </tbody>
                    </table>
                </div>

                <!-- Pagination footer -->
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 24px;">
                    <span style="font-size: 0.85rem; color: var(--text-muted);" id="pagination-info">
                        Showing page ${currentPage}
                    </span>
                    <div style="display: flex; gap: 8px;">
                        <button class="btn-secondary" id="btn-prev-page" style="padding: 8px 16px;">
                            <i data-lucide="chevron-left"></i> Previous
                        </button>
                        <button class="btn-secondary" id="btn-next-page" style="padding: 8px 16px;">
                            Next <i data-lucide="chevron-right"></i>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;

    const tbody = document.getElementById("history-table-tbody");
    const prevBtn = document.getElementById("btn-prev-page");
    const nextBtn = document.getElementById("btn-next-page");
    const paginationInfo = document.getElementById("pagination-info");
    
    const searchInput = document.getElementById("history-search-input");
    const typeSelect = document.getElementById("history-type-select");
    const outcomeSelect = document.getElementById("history-outcome-select");
    const applyFiltersBtn = document.getElementById("btn-apply-filters");

    // Load table data function
    async function loadTableData() {
        const offset = (currentPage - 1) * pageSize;
        const allRecords = await apiClient.getHistory(100, 0); // retrieve large slice to perform local filter
        
        let filtered = allRecords;

        // Apply filters
        if (selectedMediaType !== 'all') {
            filtered = filtered.filter(x => x.media_type === selectedMediaType);
        }
        if (selectedOutcome !== 'all') {
            filtered = filtered.filter(x => x.label_name === selectedOutcome);
        }
        if (searchQuery) {
            filtered = filtered.filter(x => x.filename.toLowerCase().includes(searchQuery.toLowerCase()));
        }

        // Slice for pagination page
        const paginated = filtered.slice(offset, offset + pageSize);

        if (paginated.length === 0) {
            tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 32px 0;">No matching log records found.</td></tr>`;
        } else {
            tbody.innerHTML = paginated.map(item => {
                const badgeClass = item.label_name === "FAKE" ? "badge-fake" : "badge-real";
                const formattedDate = new Date(item.created_at).toLocaleTimeString() + " — " + new Date(item.created_at).toLocaleDateString();
                return `
                    <tr>
                        <td style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-muted);">${item.id}</td>
                        <td style="font-weight: 600; color: var(--text-primary);">${item.filename}</td>
                        <td style="text-transform: uppercase; font-family: var(--font-mono); font-size: 0.75rem;">${item.media_type}</td>
                        <td><span class="badge ${badgeClass}">${item.label_name}</span></td>
                        <td style="font-family: var(--font-mono); color: var(--color-primary); font-weight: 700;">
                            ${(item.confidence * 100).toFixed(2)}%
                        </td>
                        <td>${item.faces_count}</td>
                        <td style="color: var(--text-muted); font-size: 0.85rem;">${formattedDate}</td>
                    </tr>
                `;
            }).join('');
        }

        // Update pagination UI
        const maxPages = Math.max(Math.ceil(filtered.length / pageSize), 1);
        paginationInfo.textContent = `Showing page ${currentPage} of ${maxPages} (${filtered.length} total entries)`;
        
        prevBtn.disabled = currentPage === 1;
        nextBtn.disabled = currentPage === maxPages;
    }

    // Pagination events
    prevBtn.addEventListener("click", () => {
        if (currentPage > 1) {
            currentPage--;
            loadTableData();
        }
    });

    nextBtn.addEventListener("click", () => {
        currentPage++;
        loadTableData();
    });

    // Filters application
    applyFiltersBtn.addEventListener("click", () => {
        searchQuery = searchInput.value;
        selectedMediaType = typeSelect.value;
        selectedOutcome = outcomeSelect.value;
        currentPage = 1;
        loadTableData();
    });

    // Run initial load
    await loadTableData();
    lucide.createIcons();
}
