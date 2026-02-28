output "state_machine_arn" {
  value = aws_sfn_state_machine.batch_pipeline.arn
}

output "launch_template_id" {
  value = aws_launch_template.gpu.id
}

output "input_bucket" {
  value = local.input_bucket
}

output "output_bucket" {
  value = local.output_bucket
}
