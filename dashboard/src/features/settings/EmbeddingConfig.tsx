import { useEffect, useState } from "react"
import {
  useEmbeddingConfig,
  useUpdateConfig,
  useTestEmbedding,
} from "@/api/hooks/useDashboard"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Lock } from "lucide-react"
import { toast } from "sonner"
import { useTranslation } from "react-i18next"

const PROVIDERS = [
  { value: "sentence_transformer", label: "Sentence Transformer (Local)" },
  { value: "ollama", label: "Ollama (Local)" },
  { value: "gemini", label: "Gemini (Google)" },
  { value: "openai", label: "OpenAI" },
  { value: "openrouter", label: "OpenRouter" },
] as const

type ProviderValue = (typeof PROVIDERS)[number]["value"]

const DEFAULT_MODELS: Record<ProviderValue, string> = {
  sentence_transformer: "all-MiniLM-L6-v2",
  ollama: "nomic-embed-text",
  gemini: "text-embedding-004",
  openai: "text-embedding-3-small",
  openrouter: "openai/text-embedding-3-small",
}

interface FormState {
  enabled: boolean
  provider: ProviderValue
  model: string
  similarity_threshold: number
}

const isPro = false

export default function EmbeddingConfig() {
  const { t } = useTranslation()
  const { data: serverData } = useEmbeddingConfig()
  const updateConfig = useUpdateConfig()
  const testEmbedding = useTestEmbedding()

  const [form, setForm] = useState<FormState>({
    enabled: false,
    provider: "sentence_transformer",
    model: DEFAULT_MODELS.sentence_transformer,
    similarity_threshold: 0.7,
  })

  const [dirty, setDirty] = useState(false)

  useEffect(() => {
    if (!serverData) return
    const next: FormState = {
      enabled: serverData.enabled,
      provider: (serverData.provider as ProviderValue) ?? "sentence_transformer",
      model: serverData.model,
      similarity_threshold: serverData.similarity_threshold,
    }
    setForm(next)
    setDirty(false)
  }, [serverData])

  function updateField<K extends keyof FormState>(key: K, value: FormState[K]) {
    setForm((prev) => {
      const next = { ...prev, [key]: value }
      if (key === "provider") {
        next.model = DEFAULT_MODELS[value as ProviderValue]
      }
      return next
    })
    setDirty(true)
  }

  const handleTest = () => {
    testEmbedding.mutate(undefined, {
      onSuccess: (data) => {
        if (data.status === "ok") {
          toast.success(
            t("settings.embeddingTestOk", { provider: data.provider, dimension: data.dimension }),
          )
        } else {
          toast.error(t("settings.embeddingTestFail", { error: data.error ?? "Unknown error" }))
        }
      },
      onError: () => toast.error(t("settings.embeddingTestFail", { error: "Network error" })),
    })
  }

  const handleSave = () => {
    updateConfig.mutate(
      {
        embedding: {
          enabled: form.enabled,
          provider: form.provider,
          model: form.model,
          similarity_threshold: form.similarity_threshold,
        },
      },
      {
        onSuccess: () => {
          toast.success(t("settings.embeddingSaved"))
          setDirty(false)
        },
        onError: () => toast.error(t("settings.embeddingSaveFailed")),
      },
    )
  }

  return (
    <div className="relative">
      {!isPro && (
        <div className="absolute inset-0 z-10 flex items-start justify-end p-4 pointer-events-none">
          <Badge variant="warning" className="flex items-center gap-1">
            <Lock className="size-3" aria-hidden="true" />
            {t("settings.proFeature")}
          </Badge>
        </div>
      )}

      <div className={!isPro ? "opacity-60 pointer-events-none" : undefined}>
        <Card>
          <CardHeader>
            <CardTitle>{t("settings.embeddingConfig")}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4 text-sm">
            {/* Enable toggle */}
            <label className="flex items-center gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={form.enabled}
                onChange={(e) => updateField("enabled", e.target.checked)}
                className="size-4 cursor-pointer"
              />
              <span>{t("settings.embeddingEnabled")}</span>
            </label>

            {/* Provider */}
            <div className="space-y-1">
              <label className="text-xs font-medium text-muted-foreground" htmlFor="embedding-provider">
                {t("settings.embeddingProvider")}
              </label>
              <select
                id="embedding-provider"
                value={form.provider}
                onChange={(e) => updateField("provider", e.target.value as ProviderValue)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm cursor-pointer focus:outline-none focus:ring-2 focus:ring-ring"
              >
                {PROVIDERS.map((p) => (
                  <option key={p.value} value={p.value}>
                    {p.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Model */}
            <div className="space-y-1">
              <label className="text-xs font-medium text-muted-foreground" htmlFor="embedding-model">
                {t("settings.embeddingModel")}
              </label>
              <input
                id="embedding-model"
                type="text"
                value={form.model}
                onChange={(e) => updateField("model", e.target.value)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-ring"
              />
            </div>

            {/* Similarity threshold */}
            <div className="space-y-1">
              <div className="flex items-center justify-between">
                <label className="text-xs font-medium text-muted-foreground" htmlFor="embedding-threshold">
                  {t("settings.embeddingThreshold")}
                </label>
                <span className="font-mono text-xs">{form.similarity_threshold.toFixed(2)}</span>
              </div>
              <input
                id="embedding-threshold"
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={form.similarity_threshold}
                onChange={(e) => updateField("similarity_threshold", parseFloat(e.target.value))}
                className="w-full cursor-pointer accent-primary"
              />
            </div>

            {/* Actions */}
            <div className="flex gap-2 pt-1">
              <Button
                variant="outline"
                size="sm"
                onClick={handleTest}
                disabled={testEmbedding.isPending}
                className="cursor-pointer"
              >
                {testEmbedding.isPending ? t("settings.embeddingTesting") : t("settings.embeddingTest")}
              </Button>
              <Button
                size="sm"
                onClick={handleSave}
                disabled={!dirty || updateConfig.isPending}
                className="cursor-pointer"
              >
                {updateConfig.isPending ? t("settings.embeddingSaving") : t("settings.embeddingSave")}
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
