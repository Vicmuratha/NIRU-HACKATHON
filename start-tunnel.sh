#!/bin/bash
# ─── SafEye Public Tunnel ───
# Makes your local app accessible from any phone/network
# Uses localtunnel (free, no signup)

set -e

FRONTEND_PORT=3000
BACKEND_PORT=7860
SUBDOMAIN="${1:-safeye-$(whoami)}"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  🌍 SafEye — Public Tunnel Setup"
echo "═══════════════════════════════════════════════════"
echo ""

# Check if servers are running
if ! curl -s http://localhost:$FRONTEND_PORT > /dev/null 2>&1; then
    echo "⚠️  Frontend not detected on port $FRONTEND_PORT"
    echo "   Start it first: npm run dev"
    echo ""
fi

if ! curl -s http://localhost:$BACKEND_PORT > /dev/null 2>&1; then
    echo "⚠️  Backend not detected on port $BACKEND_PORT"
    echo "   Start it first: python app.py"
    echo ""
fi

echo "🚀 Starting tunnel for frontend (port $FRONTEND_PORT)..."
echo "   Subdomain: $SUBDOMAIN"
echo ""

# Start the tunnel for the frontend (which proxies /api to backend)
npx localtunnel --port $FRONTEND_PORT --subdomain "$SUBDOMAIN" &
LT_PID=$!

sleep 3

TUNNEL_URL="https://${SUBDOMAIN}.loca.lt"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  ✅ Tunnel is LIVE!"
echo ""
echo "  📱 Open on your phone:"
echo "     $TUNNEL_URL"
echo ""
echo "  ⚠️  First visit: click 'Click to Continue' on the"
echo "     localtunnel splash page (this is normal)"
echo ""
echo "  🔑 For Google OAuth to work from the tunnel,"
echo "     update Google Cloud Console redirect URIs:"
echo "     ${TUNNEL_URL}/auth/google/callback"
echo ""
echo "  💡 Set FRONTEND_URL env var for Flask:"
echo "     export FRONTEND_URL=$TUNNEL_URL"
echo "     Then restart: python app.py"
echo ""
echo "  Press Ctrl+C to stop the tunnel"
echo "═══════════════════════════════════════════════════"
echo ""

# Wait for tunnel process
wait $LT_PID
