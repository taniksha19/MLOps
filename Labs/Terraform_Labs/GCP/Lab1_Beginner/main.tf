provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# 1. FIREWALL: Explicitly allow HTTP traffic (Port 80)
resource "google_compute_firewall" "web_firewall" {
  name    = "allow-http-traffic"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["80"]
  }

  source_ranges = ["0.0.0.0/0"] # Open to the world
  target_tags   = ["http-server"]
}

# 2. VIRTUAL MACHINE: The server itself
resource "google_compute_instance" "web_server" {
  name         = "terraform-nginx-vm"
  machine_type = var.machine_type
  zone         = var.zone
  tags         = ["http-server"] # Connects to the firewall rule above

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
    }
  }

  network_interface {
    network = "default"
    access_config {
      # Empty block assigns a Public IP
    }
  }

  # 3. PROVISIONING: Script to install Nginx automatically on boot
  metadata_startup_script = <<-EOT
    #!/bin/bash
    apt-get update
    apt-get install -y nginx
    echo '<h1>Deployed via Terraform</h1>' > /var/www/html/index.html
  EOT
}