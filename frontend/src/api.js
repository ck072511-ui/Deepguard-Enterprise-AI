/* ============================================================================
   DeepGuard Frontend API Client Handler
   ============================================================================ */

const DEFAULT_API_BASE = '/api/v1';

function getConfiguredApiBaseUrl() {
    const fromWindow = typeof window !== 'undefined' ? window.__DEEPGUARD_CONFIG__?.API_BASE_URL : null;
    if (fromWindow) return fromWindow;

    const fromEnv = typeof window !== 'undefined' ? window.__DEEPGUARD_CONFIG__?.BACKEND_URL : null;
    if (fromEnv) return fromEnv;

    return DEFAULT_API_BASE;
}

function getApiBaseUrl() {
    return getConfiguredApiBaseUrl();
}

function getApiKey() {
    if (typeof window !== 'undefined') {
        const configKey = window.__DEEPGUARD_CONFIG__?.API_KEY;
        if (configKey) return configKey;
        try {
            return localStorage.getItem('deepguard_api_key');
        } catch {
            return null;
        }
    }
    return null;
}

function buildHeaders(overrides = {}) {
    const headers = { Accept: 'application/json', ...overrides };
    const apiKey = getApiKey();
    if (apiKey) {
        headers['X-API-Key'] = apiKey;
    }
    return headers;
}

function timeoutFetch(resource, options = {}) {
    const { timeout = 15000 } = options;
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);
    const mergedOptions = { ...options, signal: controller.signal };

    return fetch(resource, mergedOptions)
        .finally(() => clearTimeout(id));
}

async function handleResponse(response) {
    if (!response.ok) {
        let body;
        try {
            body = await response.json();
        } catch {
            body = null;
        }

        const detail = body?.detail || body?.message || response.statusText || 'Unknown error';
        const error = new Error(detail);
        error.status = response.status;
        throw error;
    }

    if (response.status === 204 || response.headers.get('Content-Length') === '0') {
        return null;
    }

    return response.json();
}

async function retryRequest(requestFn, maxRetries = 2, backoffMs = 500) {
    let lastError;
    for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
        try {
            return await requestFn();
        } catch (err) {
            lastError = err;
            if (attempt === maxRetries) {
                throw err;
            }
            await new Promise((resolve) => setTimeout(resolve, backoffMs * 2 ** attempt));
        }
    }
    throw lastError;
}

export class DeepGuardAPIClient {
    constructor(baseUrl = null) {
        this.baseUrl = baseUrl || getApiBaseUrl();
    }

    async checkHealth() {
        const response = await retryRequest(() => timeoutFetch(`${this.baseUrl}/health`, {
            method: 'GET',
            headers: buildHeaders(),
            timeout: 5000,
        }), 2, 300);
        return handleResponse(response);
    }

    async getHistory(limit = 20, offset = 0) {
        const page = Math.floor(offset / limit) + 1;
        const response = await retryRequest(() => timeoutFetch(
            `${this.baseUrl}/history?page_size=${limit}&page=${page}&sort_by=created_at&order=desc`,
            {
                method: 'GET',
                headers: buildHeaders(),
                timeout: 10000,
            }
        ));
        const data = await handleResponse(response);
        return data.items || data || [];
    }

    async getStats() {
        const response = await retryRequest(() => timeoutFetch(`${this.baseUrl}/history/stats`, {
            method: 'GET',
            headers: buildHeaders(),
            timeout: 10000,
        }), 2, 300);
        return handleResponse(response);
    }

    async getCurrentUser() {
        const response = await timeoutFetch(`${this.baseUrl}/auth/me`, {
            method: 'GET',
            headers: buildHeaders(),
            timeout: 8000,
        });
        return handleResponse(response);
    }

    async listModels() {
        const response = await retryRequest(() => timeoutFetch(`${this.baseUrl}/models`, {
            method: 'GET',
            headers: buildHeaders(),
            timeout: 10000,
        }));
        return handleResponse(response) || [];
    }

    async registerModel(name, version, registryPath) {
        const response = await timeoutFetch(`${this.baseUrl}/models`, {
            method: 'POST',
            headers: buildHeaders({ 'Content-Type': 'application/json' }),
            body: JSON.stringify({ name, version, registry_path: registryPath }),
            timeout: 10000,
        });
        return handleResponse(response);
    }

    async activateModel(modelId) {
        const response = await timeoutFetch(`${this.baseUrl}/models/${modelId}/activate`, {
            method: 'POST',
            headers: buildHeaders(),
            timeout: 10000,
        });
        return handleResponse(response);
    }

    async getActiveModelInfo() {
        const response = await timeoutFetch(`${this.baseUrl}/model-info/active`, {
            method: 'GET',
            headers: buildHeaders(),
            timeout: 10000,
        });
        return handleResponse(response);
    }

    async getTrainingSummary() {
        const response = await timeoutFetch(`${this.baseUrl}/model-info/training`, {
            method: 'GET',
            headers: buildHeaders(),
            timeout: 10000,
        });
        return handleResponse(response);
    }

    async detectMedia(file) {
        const formData = new FormData();
        formData.append('file', file, file.name);

        const response = await timeoutFetch(`${this.baseUrl}/detect`, {
            method: 'POST',
            body: formData,
            headers: buildHeaders(),
            timeout: 300000,
        });
        return handleResponse(response);
    }
}
