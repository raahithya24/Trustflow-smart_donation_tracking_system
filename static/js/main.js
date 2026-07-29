// API Base URL
const API_BASE = '';

// Utility Functions
function showAlert(message, type = 'success') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.textContent = message;
    
    const container = document.querySelector('.container');
    container.insertBefore(alertDiv, container.firstChild);
    
    setTimeout(() => alertDiv.remove(), 5000);
}

function formatCurrency(amount) {
    return new Intl.NumberFormat('en-IN', {
        style: 'currency',
        currency: 'INR',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(amount);
}

function formatDate(dateString) {
    return new Date(dateString).toLocaleDateString('en-IN', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

function showLoading(show = true) {
    const loader = document.getElementById('loading-spinner') || createLoadingSpinner();
    loader.style.display = show ? 'block' : 'none';
}

function createLoadingSpinner() {
    const spinner = document.createElement('div');
    spinner.id = 'loading-spinner';
    spinner.className = 'spinner';
    spinner.style.display = 'none';
    document.body.appendChild(spinner);
    return spinner;
}

// API Calls
async function apiCall(url, method = 'GET', data = null) {
    const options = {
        method,
        headers: {
            'Content-Type': 'application/json',
        },
        credentials: 'same-origin'
    };
    
    if (data) {
        options.body = JSON.stringify(data);
    }
    
    try {
        showLoading(true);
        const response = await fetch(url, options);
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.message || 'API call failed');
        }
        
        return result;
    } catch (error) {
        showAlert(error.message, 'error');
        throw error;
    } finally {
        showLoading(false);
    }
}

// Login Function
async function login(email, password) {
    const result = await apiCall('/login', 'POST', { email, password });

    if (result.status === 'success') {
        showAlert('Login successful! Redirecting...');

        // Redirect based on role
        if (result.role === 'donor') {
            window.location.href = '/web/donor';
        } 
        else if (result.role === 'admin') {
            window.location.href = '/web/admin';
        } 
        else if (result.role === 'ngo') {
            window.location.href = '/web/ngo';
        }
        else {
            // Fallback (just in case)
            window.location.href = '/';
        }

    } else {
        showAlert(result.message, 'error');
    }
}// Logout Function
async function logout() {
    const result = await apiCall('/logout', 'GET');
    showAlert('Logged out successfully');
    window.location.href = '/web/login';
}

// Register User
async function register(userData) {
    const result = await apiCall('/register', 'POST', userData);
    
    if (result.status === 'success') {
        showAlert('Registration successful! Please login.');
        setTimeout(() => {
            window.location.href = '/web/login';
        }, 2000);
    }
}

// Register NGO
async function registerNGO(ngoData) {
    const result = await apiCall('/register_ngo', 'POST', ngoData);
    
    if (result.status === 'success') {
        showAlert('NGO registration successful! Please login.');
        setTimeout(() => {
            window.location.href = '/web/login';
        }, 2000);
    }
}