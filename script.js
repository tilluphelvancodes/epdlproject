/**
 * TITAN - Final Landing Page Script
 */

// --- 1. MOUSE PARALLAX EFFECT ---
// Moves the background image slightly in response to mouse movement for a high-end feel
document.addEventListener('mousemove', (e) => {
    const bg = document.querySelector('.bg-image');
    if (bg) {
        // Dividing by 100 keeps the movement subtle
        const x = (window.innerWidth - e.pageX * 2) / 100;
        const y = (window.innerHeight - e.pageY * 2) / 100;
        
        // Scale(1.1) ensures the image covers the edges when it moves
        bg.style.transform = `translateX(${x}px) translateY(${y}px) scale(1.1)`;
    }
});

// --- 2. SMOOTH REVEAL ON PAGE LOAD ---
// Staggers the appearance of text and cards as the page opens
window.addEventListener('DOMContentLoaded', () => {
    const heroElements = document.querySelectorAll('h1, .cta-group, .card');
    
    heroElements.forEach((el, index) => {
        setTimeout(() => {
            el.style.opacity = '1';
            el.style.transform = 'translateY(0)';
        }, 200 * index); // 200ms delay between each element
    });
});

// --- 3. LOGIN MODAL LOGIC ---
// Handles opening and closing the black & neon login box
const loginLink = document.querySelector('.login');
const modal = document.getElementById('loginModal');
const closeBtn = document.getElementById('closeModal');

if (loginLink && modal && closeBtn) {
    
    // Open modal when 'Log in' is clicked
    loginLink.addEventListener('click', (e) => {
        e.preventDefault(); // Stop page refresh
        modal.style.display = 'flex';
    });

    // Close modal when 'X' is clicked
    closeBtn.addEventListener('click', () => {
        modal.style.display = 'none';
    });

    // Close modal if user clicks outside of the box (on the blur)
    window.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.style.display = 'none';
        }
    });
}

// Optional: Prevent form from refreshing page on 'Sign In' click
const loginForm = document.querySelector('.login-box form');
if (loginForm) {
    loginForm.addEventListener('submit', (e) => {
        e.preventDefault();
        alert('Titan AI: Authentication sequence initiated...');
        // You can add redirection logic here
    });
}