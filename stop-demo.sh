#!/bin/bash

echo "🛑 Shutting down Brain Tumor Detection Demo..."

# Kill all port-forward processes
echo "📡 Stopping port forwarding..."
pkill -f "port-forward" 2>/dev/null || true

# Delete all your app resources
echo "🗑️  Deleting application resources..."
kubectl delete -k k8s/ 2>/dev/null || true

# Delete Kubeflow namespace (this removes all Kubeflow stuff)
echo "🔬 Removing Kubeflow..."
kubectl delete namespace kubeflow --ignore-not-found=true

# Stop minikube completely
echo "📦 Stopping minikube..."
minikube stop

echo ""
echo "✅ Everything shut down!"
echo "💡 Run './start-demo.sh' for a completely fresh start"
