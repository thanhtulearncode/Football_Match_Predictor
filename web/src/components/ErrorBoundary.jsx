import { Component } from "react";
import Icon from "./Icon";

/**
 * ErrorBoundary Component
 * Catches JavaScript errors anywhere in the child component tree
 */
class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("ErrorBoundary caught an error:", error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-slate-900 text-slate-200 flex items-center justify-center p-4">
          <div className="max-w-md w-full bg-slate-800/60 backdrop-blur-md border border-rose-500/20 rounded-xl p-8 text-center">
            <Icon
              name="alert-triangle"
              className="w-16 h-16 text-rose-500 mx-auto mb-4"
            />
            <h2 className="text-2xl font-bold mb-2 text-rose-400">
              Something went wrong
            </h2>
            <p className="text-slate-400 mb-6">
              {this.state.error?.message || "An unexpected error occurred"}
            </p>
            <button
              onClick={this.handleReset}
              className="px-6 py-3 bg-blue-500 hover:bg-blue-600 text-white font-bold rounded-lg transition-colors"
            >
              Try Again
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
