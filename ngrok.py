import time
import requests
from pyngrok import ngrok

# ===== CONFIGURATION =====
NGROK_PORT = 5000                    # Your app's port (change if needed)
DUCKDNS_DOMAIN = "ahmed313"          # Your DuckDNS subdomain
DUCKDNS_TOKEN = "c7707c5a-b275-4cfe-b50f-a5a02728fdbc"  # Your DuckDNS token
NGROK_AUTHTOKEN = "30SKKtxQBIHbKd6JNeFqStm0gaW_6FzmiFz9kjidKjEXioM83"  # Your Ngrok token

# ===== INITIALIZE NGROK =====
ngrok.set_auth_token(NGROK_AUTHTOKEN)

def update_duckdns(ngrok_url):
    """Update DuckDNS with the new Ngrok URL"""
    # Extract the Ngrok subdomain (e.g. "abc123" from "https://abc123.ngrok.io")
    ngrok_host = ngrok_url.replace("https://", "").split(".ngrok.io")[0]
    
    # Send update to DuckDNS
    response = requests.get(
        f"https://www.duckdns.org/update?"
        f"domains={DUCKDNS_DOMAIN}&token={DUCKDNS_TOKEN}&txt={ngrok_host}"
    )
    print(f"DuckDNS Update: {response.text}")

# ===== MAIN LOOP =====
print("Starting Ngrok+DuckDNS integration...")
while True:
    try:
        # Start Ngrok tunnel
        tunnel = ngrok.connect(NGROK_PORT, "http")
        public_url = tunnel.public_url
        print(f"\nNgrok Public URL: {public_url}")
        
        # Update DuckDNS
        update_duckdns(public_url)
        print(f"Your permanent URL: https://{DUCKDNS_DOMAIN}.duckdns.org")
        
        # Wait 8 hours (use 30 seconds for testing)
        print("Running for 8 hours (Ctrl+C to stop)...")
        time.sleep(28800)  
        
        # Restart Ngrok (free tier requires refresh every 8 hours)
        print("\nRestarting Ngrok tunnel...")
        ngrok.kill()
        
    except Exception as e:
        print(f"Error: {e}. Retrying in 5 seconds...")
        time.sleep(5)