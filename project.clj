(defproject dl4j-clj-example "0.1.0"
  :description "DL4J's Iris example straight port to Clojure"
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [org.deeplearning4j/deeplearning4j-core "0.4-rc3.8"]
                 [commons-io/commons-io "2.5"]
                 [org.nd4j/nd4j-x86 "0.4-rc3.8"]]
  :main ^:skip-aot dl4j-clj-example.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})                             ;;; Won't build correctly. See docs.
