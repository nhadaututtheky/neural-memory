import { create } from "zustand"

interface LivingBrainState {
  hoveredId: string | null
  selectedId: string | null
  setHovered: (id: string | null) => void
  setSelected: (id: string | null) => void
  clear: () => void
}

export const useLivingBrainStore = create<LivingBrainState>((set) => ({
  hoveredId: null,
  selectedId: null,
  setHovered: (id) => set({ hoveredId: id }),
  setSelected: (id) => set({ selectedId: id }),
  clear: () => set({ hoveredId: null, selectedId: null }),
}))
