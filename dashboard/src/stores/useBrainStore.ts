import { create } from "zustand"

interface BrainState {
  activeBrain: string | null
  setActiveBrain: (name: string) => void
}

export const useBrainStore = create<BrainState>((set) => ({
  activeBrain: null,
  setActiveBrain: (name) => set({ activeBrain: name }),
}))
