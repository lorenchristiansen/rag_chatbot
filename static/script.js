let lastResponse;

// Function to send a message
function sendMessage() {
    let userInput = document.getElementById("user-input").value;
    if (!userInput.trim()) return;

    // Clear the input field immediately
    document.getElementById("user-input").value = "";

    let chatBox = document.getElementById("chat-box");
    chatBox.innerHTML += `<p><strong>You:</strong> ${userInput}</p>`;

    // Send the message to the backend
    fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userInput })
    })
    .then(response => response.json())
    .then(data => {
        // Show the bot's response
        lastResponse = data;
        chatBox.innerHTML += `<p><strong>Bot:</strong> ${data}</p>`;
        chatBox.scrollTop = chatBox.scrollHeight;
        
    })
    .catch(error => console.error("Error:", error));
}


// Listen for the Enter key press
document.getElementById("user-input").addEventListener("keydown", function(event) {
    if (event.key === "Enter") {
        event.preventDefault();  // Prevent the default behavior (like submitting forms)
        sendMessage();  // Trigger sendMessage function
    }
});
