import { Component, type ReactNode } from "react"

interface Props {
  children: ReactNode
  fallback: ReactNode
  onError?: (error: Error) => void
}

interface State {
  hasError: boolean
}

export class CanvasErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false }

  static getDerivedStateFromError(): State {
    return { hasError: true }
  }

  componentDidCatch(error: Error): void {
    this.props.onError?.(error)
  }

  render() {
    if (this.state.hasError) return this.props.fallback
    return this.props.children
  }
}
