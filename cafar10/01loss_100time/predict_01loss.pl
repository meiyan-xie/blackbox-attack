$test=shift;
#$testlabels=shift;
$map=shift;

open(MAP,$map);
$| = 1;
$i=0;
while(<MAP>){

  chomp $_; @s=split(/\s+/,$_); $c1=$s[1]; $c2=$s[2];
  $pos[$i] = $c1;
  $neg[$i] = $c2;
  open(IN, "icd.cifar_mc.$i");
  $j=0;
  while(<IN>){
   if($_ =~ /Best w/){
    # if($_ =~ /Final best w/){
      $l=<IN>; chomp $l;
      $w[$i][$j] = $l;
      $l=<IN>; chomp $l;
      $w0[$i][$j] = $l;
      $j++;
    }
  }
  $acc[$i] = 1;
  $i++;
}
$boot = $j;
$total = $i;

#print "Boot=$boot Total=$total\n";

open(DATA, $test);
open(LABELS,$testlabels);

$k=0;
while(<DATA>){
  @s=split(/\s+/,$_);
  %pred=(); $pred=-1;
  for(my $i=0; $i<$total; $i++){
    $pos=0; $neg=0;
    for(my $k=0; $k<$boot; $k++){
      @w_ = split(/\s+/,$w[$i][$k]);
      $w0_ = $w0[$i][$k];
      $dp=0;
      for(my $j=0; $j<@w_; $j++){
        $dp += $s[$j] * $w_[$j];
      }
      $dp += $w0_;
#      print "dp=$dp\n";
      if($dp > 0) { $pos++; } else { $neg++; }
    }
    if($pos > $neg) { $pred{$pos[$i]} += $acc[$i]; } else { $pred{$neg[$i]} += $acc[$i]; }
  }
  @sorted_keys = sort { $pred{$b} <=> $pred{$a} } keys(%pred);
#  for(my $i=0; $i<@sorted_keys; $i++){ print "$i $sorted_keys[$i] $pred{$sorted_keys[$i]}\n"; }
  $pred=$sorted_keys[0];
#  print "maxdp=$maxdp pred=$pred\n";
  $truelabel=<LABELS>; chomp $truelabel; $truelabel =~ s/\cM//g;
  print "$pred\n";
#  print "k=$k\n";
  if($pred!=$truelabel) { $err++; }
  $k++;
  $error = $err/$k;
#  print "$flag error=$error\n";
}

#print "$err $k \n";
