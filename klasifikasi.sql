-- phpMyAdmin SQL Dump
-- version 4.5.1
-- http://www.phpmyadmin.net
--
-- Host: 127.0.0.1
-- Generation Time: Jul 22, 2020 at 01:10 PM
-- Server version: 10.1.19-MariaDB
-- PHP Version: 7.0.13

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `klasifikasi`
--

-- --------------------------------------------------------

--
-- Table structure for table `uji`
--

CREATE TABLE `uji` (
  `id_uji` int(11) NOT NULL,
  `label_pred` varchar(10) CHARACTER SET latin1 NOT NULL,
  `id_web` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

--
-- Dumping data for table `uji`
--

INSERT INTO `uji` (`id_uji`, `label_pred`, `id_web`) VALUES
(1, 'anger', 4),
(2, 'anger', 5),
(3, 'happy', 6),
(4, 'anger', 7),
(5, 'happy', 8),
(6, 'anger', 9),
(7, 'happy', 10),
(8, 'sadness', 11),
(9, 'anger', 12),
(10, 'sadness', 13),
(11, 'sadness', 18),
(12, 'happy', 75),
(18, 'happy', 77),
(19, 'happy', 78),
(20, 'sadness', 79);

-- --------------------------------------------------------

--
-- Table structure for table `web`
--

CREATE TABLE `web` (
  `id` int(11) NOT NULL,
  `isi_tw` varchar(300) CHARACTER SET latin1 NOT NULL,
  `hasil_pre` text NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

--
-- Dumping data for table `web`
--

INSERT INTO `web` (`id`, `isi_tw`, `hasil_pre`) VALUES
(4, 'Soal jln Jatibaru,polisi tdk bs GERTAK gubernur .Emangny polisi tdk ikut pmbhasan? Jgn berpolitik. Pengaturan wilayah,hak gubernur. Persoalan Tn Abang soal turun temurun.Pelik.Perlu kesabaran. [USERNAME] [USERNAME] [URL]', '[''soal'', ''jln'', ''jatibaru'', ''polisi'', ''tdk'', ''bs'', ''gertak'', ''gubernur'', ''emangny'', ''polisi'', ''tdk'', ''ikut'', ''pmbhasan'', ''jgn'', ''politik'', ''atur'', ''wilayah'', ''hak'', ''gubernur'', ''soal'', ''tn'', ''abang'', ''soal'', ''turun'', ''turun'', ''pelik'', ''perlu'', ''sabar'', ''username'', ''username'', ''url'']'),
(5, 'Sesama cewe lho (kayaknya), harusnya bisa lebih rasain lah yang harus sibuk jaga diri, rasain sakitnya haid, dan paniknya pulang malem sendirian. Gimana orang asing? Wajarlah banyak korban yang takut curhat, bukan dibela malah dihujat.', '[''sama'', ''cewe'', ''lho'', ''kayak'', ''harus'', ''bisa'', ''lebih'', ''rasain'', ''lah'', ''yang'', ''harus'', ''sibuk'', ''jaga'', ''diri'', ''rasain'', ''sakit'', ''haid'', ''dan'', ''panik'', ''pulang'', ''malem'', ''sendiri'', ''gimana'', ''orang'', ''asing'', ''wajar'', ''banyak'', ''korban'', ''yang'', ''takut'', ''curhat'', ''bukan'', ''bela'', ''malah'', ''hujat'']'),
(6, 'Kepingin gudeg mbarek Bu hj. Amad Foto dari google, sengaja, biar teman-teman jg membayangkannya. Berbagi itu indah.', '[''kepingin'', ''gudeg'', ''mbarek'', ''bu'', ''hj'', ''amad'', ''foto'', ''dari'', ''google'', ''sengaja'', ''biar'', ''teman'', ''jg'', ''bayang'', ''bagi'', ''itu'', ''indah'']'),
(7, 'Jln Jatibaru,bagian dari wilayah Tn Abang.Pengaturan wilayah tgg jwb dan wwnang gub.Tng Abng soal rumit,sejak gub2 , trdahulu.Skrg sedng dibenahi,agr bermnfaat semua pihak.Mohon yg punya otak,berpikirlah dgn wajar,kecuali otaknya butek.Ya kamu. [URL]', '[''jln'', ''jatibaru'', ''bagi'', ''dari'', ''wilayah'', ''tn'', ''abang'', ''atur'', ''wilayah'', ''tgg'', ''jwb'', ''dan'', ''wwnang'', ''gub'', ''tng'', ''abng'', ''soal'', ''rumit'', ''sejak'', ''gub'', ''trdahulu'', ''skrg'', ''sedng'', ''benah'', ''agr'', ''bermnfaat'', ''semua'', ''pihak'', ''mohon'', ''yg'', ''punya'', ''otak'', ''pikir'', ''dgn'', ''wajar'', ''kecuali'', ''otak'', ''butek'', ''ya'', ''kamu'', ''url'']'),
(8, 'Sharing pengalaman aja, kemarin jam 18.00 batalin tiket di stasiun pasar senen, lancar, antrian tidak terlalu rame,15 menitan dan beress semua! Mungkin bisa dicoba twips, di jam-jam segitu  cc [USERNAME]', '[''sharing'', ''alam'', ''aja'', ''kemarin'', ''jam'', ''batalin'', ''tiket'', ''di'', ''stasiun'', ''pasar'', ''senen'', ''lancar'', ''antri'', ''tidak'', ''terlalu'', ''rame'', ''menit'', ''dan'', ''beress'', ''semua'', ''mungkin'', ''bisa'', ''coba'', ''twips'', ''di'', ''jam'', ''segitu'', ''cc'', ''username'']'),
(9, 'Dari sekian banyak thread yang aku baca, thread ini paling aneh sih dalam penulisan. Sumpah aneh bgt, mau ngatain "lebay" aja segala bikin thread hadeh. Aku juga ga jago nulis, tapi tulisan aku ga seberantakan thread mbaknya.', '[''dari'', ''sekian'', ''banyak'', ''thread'', ''yang'', ''aku'', ''baca'', ''thread'', ''ini'', ''paling'', ''aneh'', ''sih'', ''dalam'', ''tulis'', ''sumpah'', ''aneh'', ''bgt'', ''mau'', ''ngatain'', ''lebay'', ''aja'', ''segala'', ''bikin'', ''thread'', ''hadeh'', ''aku'', ''juga'', ''ga'', ''jago'', ''nulis'', ''tapi'', ''tulis'', ''aku'', ''ga'', ''beranta'', ''thread'', ''mbak'']'),
(10, 'Sharing sama temen tuh emg guna bgt. Disaat lu ngerasa masalah lu berat bgt ternyata temen kita sendiri punya masalah lebih berat dr kita. Malah masalah kita ngga ada apa2nya dibanding masalah dia.', '[''sharing'', ''sama'', ''temen'', ''tuh'', ''emg'', ''guna'', ''bgt'', ''saat'', ''lu'', ''ngerasa'', ''masalah'', ''lu'', ''berat'', ''bgt'', ''nyata'', ''temen'', ''kita'', ''sendiri'', ''punya'', ''masalah'', ''lebih'', ''berat'', ''dr'', ''kita'', ''malah'', ''masalah'', ''kita'', ''ngga'', ''ada'', ''apa'', ''banding'', ''masalah'', ''dia'']'),
(11, 'Orang lain kalau pake ponco itu buat jas hujan, nah dia pake buat kasur. Ya tadi gara2 saking gak punya apa2. Mamak bilang, kami tuh di awal pernikan gak ada ngalamin bulan madu kayak skrg2. Org tidur nya aja pake ponco. Gimane mau bulan madu.', '[''orang'', ''lain'', ''kalau'', ''pake'', ''ponco'', ''itu'', ''buat'', ''jas'', ''hujan'', ''nah'', ''dia'', ''pake'', ''buat'', ''kasur'', ''ya'', ''tadi'', ''gara'', ''saking'', ''gak'', ''punya'', ''apa'', ''mamak'', ''bilang'', ''kami'', ''tuh'', ''di'', ''awal'', ''pernik'', ''gak'', ''ada'', ''ngalamin'', ''bulan'', ''madu'', ''kayak'', ''skrg'', ''org'', ''tidur'', ''nya'', ''aja'', ''pake'', ''ponco'', ''gimane'', ''mau'', ''bulan'', ''madu'']'),
(12, 'Contoh mereka yg gemar menyudutkan, teriak paling toleran tp gemar menuduh, gemar men-judge seseorang berdasarkan versi mereka. Cukup tau saja.', '[''contoh'', ''mereka'', ''yg'', ''gemar'', ''sudut'', ''teriak'', ''paling'', ''toleran'', ''tp'', ''gemar'', ''tuduh'', ''gemar'', ''men-judge'', ''orang'', ''dasar'', ''versi'', ''mereka'', ''cukup'', ''tau'', ''saja'']'),
(13, 'Pulang udah H-4 lebaran dilema sekali. Seperti tidak bisa melakukan apa2 dirumah sebelum lebaran. Buka puasa bareng cuman 3 hari sama keluarga begitu juga sahur.', '[''pulang'', ''udah'', ''h-'', ''lebaran'', ''dilema'', ''sekali'', ''seperti'', ''tidak'', ''bisa'', ''laku'', ''apa'', ''rumah'', ''belum'', ''lebaran'', ''buka'', ''puasa'', ''bareng'', ''cuman'', ''hari'', ''sama'', ''keluarga'', ''begitu'', ''juga'', ''sahur'']'),
(18, 'akhirnya yah... bisa tidur dengan tenang. walopun cuma sedikit sih', '[''akhir'', ''yah'', ''bisa'', ''tidur'', ''dengan'', ''tenang'', ''walopun'', ''cuma'', ''sedikit'', ''sih'']'),
(75, 'makan dikit tp cepat gemuk, aku makan banyak, kerjaannya ngemil tp tetap kurus', '[''makan'', ''dikit'', ''tp'', ''cepat'', ''gemuk'', ''aku'', ''makan'', ''banyak'', ''kerja'', ''ngemil'', ''tp'', ''tetap'', ''kurus'']'),
(77, 'meratapi nasib jelas bukan solusi ', '[''ratap'', ''nasib'', ''jelas'', ''bukan'', ''solusi'']'),
(78, 'sedih banget sendirian doang disini', '[''sedih'', ''banget'', ''sendiri'', ''doang'', ''sini'']'),
(79, 'Lanjut ngerajut, ni mata udh muter2 kecapekan semingguan belajarin benang.', '[''lanjut'', ''ngerajut'', ''ni'', ''mata'', ''udh'', ''muter'', ''cape'', ''minggu'', ''belajarin'', ''benang'']');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `uji`
--
ALTER TABLE `uji`
  ADD PRIMARY KEY (`id_uji`),
  ADD KEY `id_web` (`id_web`);

--
-- Indexes for table `web`
--
ALTER TABLE `web`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `uji`
--
ALTER TABLE `uji`
  MODIFY `id_uji` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=21;
--
-- AUTO_INCREMENT for table `web`
--
ALTER TABLE `web`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=80;
--
-- Constraints for dumped tables
--

--
-- Constraints for table `uji`
--
ALTER TABLE `uji`
  ADD CONSTRAINT `uji_ibfk_1` FOREIGN KEY (`id_web`) REFERENCES `web` (`id`) ON DELETE CASCADE ON UPDATE CASCADE;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
