var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl);
});

// When the page is scrolled
window.onscroll = function() {
    // Get the navbar element
    var navbar = document.querySelector('.navbar');
    
    // Check the scroll position and adjust opacity
    if (window.scrollY > 50) { // When scroll position is more than 50px
        navbar.style.opacity = 0.7; // Decrease opacity
    } else {
        navbar.style.opacity = 1; // Reset opacity
    }
};




