import { create } from "zustand"
import { persist, createJSONStorage } from "zustand/middleware"

export interface BrainSettingsState {
  effects: boolean
  brainShell: boolean
  activationPulse: boolean
  labels: boolean
  setEffects: (v: boolean) => void
  setBrainShell: (v: boolean) => void
  setActivationPulse: (v: boolean) => void
  setLabels: (v: boolean) => void
}

const STORAGE_KEY = "nm.livingBrain.settings"

export const useBrainSettings = create<BrainSettingsState>()(
  persist(
    (set) => ({
      effects: true,
      brainShell: true,
      activationPulse: true,
      labels: false,
      setEffects: (v) => set({ effects: v }),
      setBrainShell: (v) => set({ brainShell: v }),
      setActivationPulse: (v) => set({ activationPulse: v }),
      setLabels: (v) => set({ labels: v }),
    }),
    {
      name: STORAGE_KEY,
      storage: createJSONStorage(() => localStorage),
      // Bump `version` when the persisted shape changes. An empty `migrate`
      // discards mismatches safely — users fall back to defaults instead of
      // resurrecting stale keys.
      version: 1,
      migrate: (state, version) => {
        if (version === 1) return state
        return undefined
      },
      partialize: (s) => ({
        effects: s.effects,
        brainShell: s.brainShell,
        activationPulse: s.activationPulse,
        labels: s.labels,
      }),
    },
  ),
)
