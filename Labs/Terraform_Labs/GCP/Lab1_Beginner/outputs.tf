output "web_server_public_ip" {
  description = "The public IP of the web server"
  value       = google_compute_instance.web_server.network_interface[0].access_config[0].nat_ip
}

output "website_url" {
  description = "Clickable link to the website"
  value       = "http://${google_compute_instance.web_server.network_interface[0].access_config[0].nat_ip}"
}