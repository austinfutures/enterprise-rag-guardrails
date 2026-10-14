# Corvanta Industries Information Security Policy

Policy ID: POL-SEC-001. Version 5.0. Effective March 1, 2026. Policy owner: Chief Information Security Officer.

## 1. Data Classification

Corvanta classifies information into four levels: Public, Internal, Confidential and Restricted. Restricted data includes customer personally identifiable information, payment card data, employee health records and merger and acquisition documents. Restricted data must be encrypted at rest and in transit and may only be stored in approved systems listed in the data inventory. Confidential data includes financial forecasts, source code and customer contracts.

## 2. Authentication and Passwords

Passwords must be at least 14 characters long. Multi-factor authentication is mandatory for all remote access, VPN connections and SaaS applications. Passwords are not rotated on a fixed schedule and must only be changed when compromise is suspected, in line with NIST SP 800-63B guidance. Shared or generic accounts are prohibited except for documented break-glass accounts.

## 3. Endpoint Security

All company laptops must use full-disk encryption with BitLocker on Windows or FileVault on macOS. Endpoint detection and response software must remain installed and active at all times. Operating system security patches must be installed within 14 days of release, and critical patches within 72 hours.

## 4. Incident Reporting

Employees must report any suspected security incident to the Security Operations Center within 1 hour of discovery. Incidents can be reported through the SOC hotline at extension 4400 or the sec-incidents channel. Lost or stolen devices must be reported within the same 1 hour window. Employees must not attempt to investigate or remediate an incident on their own.

## 5. Security Awareness Training

All employees must complete security awareness training at onboarding and annually thereafter. Phishing simulation exercises are run quarterly. The company target is a 90 percent completion rate for quarterly phishing training. Employees who fail three consecutive phishing simulations are enrolled in mandatory remedial training.

## 6. Access Reviews

Access to privileged accounts is reviewed quarterly by system owners. Standard user access is reviewed semi-annually. Access for departing employees must be revoked within 24 hours of their final working day.

## 7. Generative AI Usage

Restricted and Confidential data must never be entered into public generative AI tools. Employees may only use approved enterprise AI tools, such as Corvanta Copilot, for work involving Internal or Confidential data. Output from generative AI tools must be reviewed by a human before it is shared with customers.

## 8. Logging, Retention and Vendors

Security logs are retained for 400 days. Third-party vendors that process Restricted data must provide a current SOC 2 Type II report before contract signature. Vendor security assessments are repeated every 12 months for critical vendors.
