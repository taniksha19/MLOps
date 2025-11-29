variable "project_id" {
  description = "The GCP Project ID"
  type        = string
  # No default value -> forces you to input it, preventing accidents
}

variable "region" {
  description = "GCP Region"
  default     = "us-central1"
}

variable "zone" {
  description = "GCP Zone"
  default     = "us-central1-a"
}

variable "machine_type" {
  description = "The VM instance type"
  default     = "e2-micro"
}