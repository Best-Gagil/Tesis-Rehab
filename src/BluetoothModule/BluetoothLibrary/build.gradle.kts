plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android") // Se agrega el plugin de Kotlin
}

android {
    namespace = "com.ejemplo.bluetooth"
    compileSdk = 35

    defaultConfig {
        minSdk = 29
        targetSdk = 34 // Se recomienda 34 en lugar de 35 para evitar posibles cambios aún no documentados
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

repositories {
    google()
    mavenCentral()
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)

    //implementation("androidx.core:core-ktx:1.12.0")
    implementation("org.jetbrains.kotlin:kotlin-stdlib:1.9.23") // Asegurar que esté presente
    //implementation("androidx.annotation:annotation:1.7.1")

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}
