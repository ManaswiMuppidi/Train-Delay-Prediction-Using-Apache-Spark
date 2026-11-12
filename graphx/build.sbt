
ThisBuild / scalaVersion := "2.12.21"

ThisBuild / version := "1.0"

ThisBuild / organization := "com.traindelay"

lazy val root = (project in file("."))
  .settings(
    name := "TrainDelayGraphX",

    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % "3.5.9",
      "org.apache.spark" %% "spark-graphx" % "3.5.9",
      "org.apache.spark" %% "spark-sql" % "3.5.9"
    ),

    Compile / run / fork := true,

    Compile / run / javaOptions ++= Seq(
      "--add-exports=java.base/sun.nio.ch=ALL-UNNAMED",
      "--add-opens=java.base/java.nio=ALL-UNNAMED"
    )
  )