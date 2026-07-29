-- MySQL dump 10.13  Distrib 8.0.45, for Win64 (x86_64)
--
-- Host: 127.0.0.1    Database: smart_donation
-- ------------------------------------------------------
-- Server version	9.3.0

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `ai_metrics`
--

DROP TABLE IF EXISTS `ai_metrics`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `ai_metrics` (
  `campaign_id` int NOT NULL,
  `priority_score` float DEFAULT '0',
  `fraud_score` float DEFAULT '0',
  `predicted_shortage_flag` tinyint(1) DEFAULT '0',
  PRIMARY KEY (`campaign_id`),
  CONSTRAINT `ai_metrics_ibfk_1` FOREIGN KEY (`campaign_id`) REFERENCES `campaigns` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `ai_metrics`
--

LOCK TABLES `ai_metrics` WRITE;
/*!40000 ALTER TABLE `ai_metrics` DISABLE KEYS */;
/*!40000 ALTER TABLE `ai_metrics` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `campaign_updates`
--

DROP TABLE IF EXISTS `campaign_updates`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `campaign_updates` (
  `id` int NOT NULL AUTO_INCREMENT,
  `campaign_id` int DEFAULT NULL,
  `update_text` text,
  `proof_document` varchar(255) DEFAULT NULL,
  `update_date` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `campaign_id` (`campaign_id`),
  CONSTRAINT `campaign_updates_ibfk_1` FOREIGN KEY (`campaign_id`) REFERENCES `campaigns` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `campaign_updates`
--

LOCK TABLES `campaign_updates` WRITE;
/*!40000 ALTER TABLE `campaign_updates` DISABLE KEYS */;
/*!40000 ALTER TABLE `campaign_updates` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `campaigns`
--

DROP TABLE IF EXISTS `campaigns`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `campaigns` (
  `id` int NOT NULL AUTO_INCREMENT,
  `ngo_id` int DEFAULT NULL,
  `title` varchar(200) NOT NULL,
  `description` text,
  `category` varchar(100) DEFAULT NULL,
  `target_amount` float NOT NULL,
  `collected_amount` float DEFAULT '0',
  `urgency_level` int DEFAULT NULL,
  `severity_score` int DEFAULT NULL,
  `deadline` date DEFAULT NULL,
  `status` enum('pending','approved','rejected','completed') DEFAULT 'pending',
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `is_flagged` tinyint(1) DEFAULT '0',
  `days_active` int DEFAULT '0',
  `total_donors` int DEFAULT '0',
  `completion_rate` float DEFAULT '0',
  `priority_score` float DEFAULT '0',
  PRIMARY KEY (`id`),
  KEY `ngo_id` (`ngo_id`),
  CONSTRAINT `campaigns_ibfk_1` FOREIGN KEY (`ngo_id`) REFERENCES `ngos` (`id`) ON DELETE CASCADE,
  CONSTRAINT `campaigns_chk_1` CHECK ((`urgency_level` between 1 and 10)),
  CONSTRAINT `campaigns_chk_2` CHECK ((`severity_score` between 1 and 10))
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `campaigns`
--

LOCK TABLES `campaigns` WRITE;
/*!40000 ALTER TABLE `campaigns` DISABLE KEYS */;
INSERT INTO `campaigns` VALUES (3,3,'Medical Emergency','Urgent surgery required','Medical',500000,5000,9,10,'2026-12-30','pending','2026-02-26 16:37:15',0,0,1,1,27.4),(4,3,'Emergency Child Surgery','Urgent medical help required','Medical',300000,0,9,8,'2026-12-30','pending','2026-02-26 16:38:32',0,0,0,0,0);
/*!40000 ALTER TABLE `campaigns` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `donations`
--

DROP TABLE IF EXISTS `donations`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `donations` (
  `id` int NOT NULL AUTO_INCREMENT,
  `donor_id` int DEFAULT NULL,
  `campaign_id` int DEFAULT NULL,
  `amount` float NOT NULL,
  `donation_date` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `status` enum('success','failed') DEFAULT 'success',
  PRIMARY KEY (`id`),
  KEY `donor_id` (`donor_id`),
  KEY `campaign_id` (`campaign_id`),
  CONSTRAINT `donations_ibfk_1` FOREIGN KEY (`donor_id`) REFERENCES `users` (`id`) ON DELETE CASCADE,
  CONSTRAINT `donations_ibfk_2` FOREIGN KEY (`campaign_id`) REFERENCES `campaigns` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=9 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `donations`
--

LOCK TABLES `donations` WRITE;
/*!40000 ALTER TABLE `donations` DISABLE KEYS */;
INSERT INTO `donations` VALUES (8,11,3,5000,'2026-02-26 16:43:52','success');
/*!40000 ALTER TABLE `donations` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `fraud_logs`
--

DROP TABLE IF EXISTS `fraud_logs`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `fraud_logs` (
  `id` int NOT NULL AUTO_INCREMENT,
  `campaign_id` int DEFAULT NULL,
  `donor_id` int DEFAULT NULL,
  `reason` varchar(255) DEFAULT NULL,
  `detected_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `fraud_logs`
--

LOCK TABLES `fraud_logs` WRITE;
/*!40000 ALTER TABLE `fraud_logs` DISABLE KEYS */;
INSERT INTO `fraud_logs` VALUES (1,1,1,'Large donation spike (>50% of target)','2026-02-26 06:01:11');
/*!40000 ALTER TABLE `fraud_logs` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `ngos`
--

DROP TABLE IF EXISTS `ngos`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `ngos` (
  `id` int NOT NULL AUTO_INCREMENT,
  `user_id` int DEFAULT NULL,
  `document_score` float DEFAULT '0',
  `update_score` float DEFAULT '0',
  `completion_rate` float DEFAULT '0',
  `transparency_score` float DEFAULT '0',
  `is_verified` tinyint(1) DEFAULT '0',
  PRIMARY KEY (`id`),
  KEY `user_id` (`user_id`),
  CONSTRAINT `ngos_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `ngos`
--

LOCK TABLES `ngos` WRITE;
/*!40000 ALTER TABLE `ngos` DISABLE KEYS */;
INSERT INTO `ngos` VALUES (3,12,0,0,0,0,0);
/*!40000 ALTER TABLE `ngos` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `users`
--

DROP TABLE IF EXISTS `users`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `users` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(100) NOT NULL,
  `email` varchar(150) NOT NULL,
  `password` varchar(255) NOT NULL,
  `role` enum('donor','ngo','admin') NOT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `email` (`email`)
) ENGINE=InnoDB AUTO_INCREMENT=13 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `users`
--

LOCK TABLES `users` WRITE;
/*!40000 ALTER TABLE `users` DISABLE KEYS */;
INSERT INTO `users` VALUES (1,'Admin','admin@gmail.com','admin123','admin','2026-02-25 17:40:21'),(4,'Test Donor','donor@gmail.com','donor123','donor','2026-02-26 08:32:20'),(9,'Admin','admin_test@gmail.com','scrypt:32768:8:1$19HUKeGnpQp4fjsg$8ca7bac6ed974cce143323297ddbd7cbcd0b5416722e03aa99b5c657ef79134c0217a4a7f169df2db9ff6e6f64b20f34b96dc010877674657f2bcb008f3261ef','admin','2026-02-26 16:21:39'),(11,'Test Donor','donor_test@gmail.com','scrypt:32768:8:1$vAbdYHvuWIr2itlf$72c1ff43b3b86c04aafaa1d41944a9c5e653d95741c93b23cd6585af0099b87ceb8537a75297e8541658e51c7a6109205c14a6814af2ca568582f7ebf5628d9b','donor','2026-02-26 16:22:41'),(12,'Helping NGO','ngo_test@gmail.com','scrypt:32768:8:1$Ex20lfDW8w7EM7Ri$4ce93ca1a6cb61fcaeb565663078f62f3df7b42a0fd27e1f23abc317708a755869c46a3440339ea82f4aece15cdbda665c37b378a29b5cf8612a2e118592d211','ngo','2026-02-26 16:36:33');
/*!40000 ALTER TABLE `users` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2026-02-27  9:04:13
