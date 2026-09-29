{{/*
Expand the name of the chart.
*/}}
{{- define "laya-decision-studio.name" -}}
{{- default (default .Chart.Name .Release.Name) .Values.studio.name | trunc 63 | trimSuffix "-" }}
{{- end }}


{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "laya-decision-studio.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}


{{/*
Common labels
*/}}
{{- define "laya-decision-studio.labels" -}}
helm.sh/chart: {{ include "laya-decision-studio.chart" . }}
{{ include "laya-decision-studio.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}


{{/*
Selector labels
*/}}
{{- define "laya-decision-studio.selectorLabels" -}}
app: {{ include "laya-decision-studio.name" . }}
app.kubernetes.io/name: {{ include "laya-decision-studio.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/part-of: {{ include "laya-decision-studio.name" . }}
{{- end }}


{{/*
Create the name of the service account to use
*/}}
{{- define "laya-decision-studio.serviceAccountName" -}}
{{- if .Values.studio.serviceAccount.create }}
{{- default (include "laya-decision-studio.name" .) .Values.studio.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.studio.serviceAccount.name }}
{{- end }}
{{- end }}


{{/*
The merged environment for the app: the chart's own LAYA_DEVICE default
(cuda with the GPU enabled, cpu without) under the user's env values, so a
value set in .Values.studio.env wins.
*/}}
{{- define "laya-decision-studio.env" -}}
{{- $env := deepCopy (.Values.studio.env | default dict) -}}
{{- if .Values.studio.gpu.enabled -}}
{{- $_ := set $env "LAYA_DEVICE" (default "cuda" (get $env "LAYA_DEVICE")) -}}
{{- else -}}
{{- $_ := set $env "LAYA_DEVICE" (default "cpu" (get $env "LAYA_DEVICE")) -}}
{{- end -}}
{{- range $key, $value := $env }}
- name: {{ $key }}
  value: {{ $value | quote }}
{{- end }}
{{- end }}
